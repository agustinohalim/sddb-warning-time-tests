"""Artikel EKG 1 — paket D sesudah telaah panel 26 Oktober 2026 (Telaah/2026-10-26-EKG1/report.md).

Isu yang dijawab (id temuan panel):
  meth M1, DA X1, DA X2   perubahan buatan diskalakan dengan SD gabungan tiga basis data (memuat ragam antarpasien
                          dan antarbasis data), sehingga keunggulan uji dalam-pasien sebagian dibangun oleh skala itu.
  meth M2, DA X3          uji primer (random forest lawan kontrol pergeseran pasien sama, 15 pasien) tanpa taksiran
                          daya, sehingga p 0,189 tidak bisa dibaca sebagai "tidak ada perubahan".
  DA (tak ternilai)       ragam antarpasien lawan dalam-pasien tiap ciri tidak dilaporkan.

Bagian (argumen; tanpa argumen semua, urut):
  ragam     dekomposisi ragam tiap ciri: SD gabungan (dasar skala lama), SD dalam-rekaman gabungan (akar rerata
            ragam per rekaman, semua rekaman tiga basis data), rasionya, dan ICC = ragam antarrekaman / total.
  daya_w    simulasi daya 14_paket_b.daya dengan arah yang sama (searah_k, tegak_k) tetapi diskalakan SD
            dalam-rekaman: sel searah_w 1, 2, 4 dan tegak_w 2, 200 simulasi tiap sel (benih 5000 + k, berpasangan
            dengan simulasi nol k di b_daya.csv). Ambang dan ukuran dari simulasi nol b_daya.csv (nol tidak
            bergantung pada skala). Statistik dan uji sama dengan Tabel 4 naskah.
  primer    daya bersyarat uji primer pada pasien nyata: 15 pasien uji primer (rekaman >= 240 menit sebelum VF),
            random forest, 10 ulangan CV seperti 14_paket_b.geser. Tiap sel perubahan: 100 simulasi kontras
            teramati (suntikan dengan benih 7000 + k) dan 100 undian pergeseran, masing-masing dengan suntikannya
            sendiri (benih 8000 + k), sehingga kontaminasi jendela pergeseran oleh perubahan yang meluruh ikut
            dihitung. Tolak bila kontras teramati > persentil 95 sebaran pergeseran sel itu (setara p <= 0,05
            satu sisi). Sel nol (tanpa suntikan) memeriksa bahwa susunan ini mengulang p data nyata. Bersyarat:
            pasien tidak diambil ulang, jadi ini daya uji seperti dijalankan pada pasien ini.
Keluaran: keluaran/d_ragam.csv, d_daya_w.csv, d_primer.csv, d.log (lewat >)
Pakai:    PYTHONPATH=. python -u 15_paket_d.py [bagian ...] > keluaran/d.log
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
x = import_module('09_revisi_panel')
y10 = import_module('10_daya_ulang')
b = import_module('14_paket_b')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
JOBS = 20
SEL_W = [('searah_w', 1.0), ('searah_w', 2.0), ('searah_w', 4.0), ('tegak_w', 2.0)]
SEL_P = [('nol', 0.0), ('searah_k', 0.5), ('searah_k', 1.0), ('searah_k', 2.0), ('searah_w', 2.0),
         ('searah_w', 4.0), ('nadi', 10.0), ('nadi', 20.0)]
N_W, N_P, ULANG_P = 200, 100, 10


# ---------------------------------------------------------------- ragam
def sd_dalam(f):
    """SD dalam-rekaman gabungan: akar rerata berbobot (n - 1) ragam per rekaman, semua basis data."""
    g = f.groupby(['db', 'rec'])[CIRI]
    v, n = g.var(ddof=1), g.size() - 1
    return np.sqrt((v.mul(n, axis=0)).sum() / n.sum())


def ragam():
    f = x.fitur()
    sd = f[CIRI].std()
    sw = sd_dalam(f)
    t = pd.DataFrame({'sd_gabungan': sd, 'sd_dalam': sw, 'rasio': sw / sd, 'icc': 1 - (sw / sd) ** 2})
    t.to_csv(os.path.join(OUT, 'd_ragam.csv'))
    print('== Dekomposisi ragam tiap ciri (semua jendela tiga basis data)')
    print(t.to_string(float_format=lambda v: f'{v:.3f}'))
    arah = arah_w(f)
    for k in ('searah_k', 'tegak_k'):
        print(f'  panjang vektor {k}: skala gabungan {np.linalg.norm(arah[k] / sd.to_numpy()):.2f} SD gabungan; '
              f'{k[:-2]}_w = arah sama dalam SD dalam-rekaman, panjang dalam SD gabungan '
              f'{np.linalg.norm(arah[k[:-2] + "_w"] / sd.to_numpy()):.2f}', flush=True)


def arah_w(f0):
    """Arah 10_daya_ulang.arah_dasar ditambah varian _w: arah satuan yang sama, diskalakan SD dalam-rekaman."""
    arah = y10.arah_dasar(f0)
    sd = f0[CIRI].std().to_numpy()
    sw = sd_dalam(f0).to_numpy()
    for k in ('searah_k', 'tegak_k'):
        e = arah[k] / sd                     # arah satuan (norma 1) dalam satuan SD gabungan
        arah[k[:-2] + '_w'] = e * sw
    return arah


# ---------------------------------------------------------------- daya_w
def daya_w():
    f0, _, tetap, nsr_ids, chf_ids = b.siapkan_sim()
    arah = arah_w(f0)
    print(f'\n== Simulasi daya, perubahan dalam SD dalam-rekaman: {len(tetap)} pasien kohort tetap', flush=True)
    tugas = [(j, s, 5000 + k) for j, s in SEL_W for k in range(N_W)]
    path = os.path.join(OUT, 'd_daya_w.csv')
    b.jalankan(path, tugas, b.satu_sim, f0, arah, tetap, nsr_ids, chf_ids)
    ringkas_w()


def ringkas_w():
    h0 = pd.read_csv(os.path.join(OUT, 'b_daya.csv'))
    nol, segar = h0[h0.jenis == 'nol'].reset_index(drop=True), h0[h0.jenis == 'segar']
    h = pd.read_csv(os.path.join(OUT, 'd_daya_w.csv'))
    r = np.random.default_rng(31)
    print(f'\n== Ringkasan daya (SD dalam-rekaman): ambang dari {len(nol)} simulasi nol b_daya.csv, ukuran pada '
          f'{len(segar)} nol segar; selang 95 % bootstrap nol dan sel bersama (1.000 undian)')
    for kol, nama in (('h_nsr', 'horizon lawan nsrdb'), ('h_chf', 'horizon lawan chfdb'),
                      ('hp_nsr', 'horizon berpasangan lawan nsrdb'), ('d_auc', 'dalam-pasien AUC jendela')):
        lo, hi = b.ambang(nol, kol, 2)
        print(f'\n  {nama} [{kol}]: ambang ({lo:+.3f}, {hi:+.3f}); ukuran {b.laju(segar, kol, lo, hi):.3f}')
        for j, s in SEL_W:
            g = h[(h.jenis == j) & (h.besar == s)].reset_index(drop=True)
            bs = [b.laju(g.iloc[r.integers(0, len(g), len(g))], kol,
                         *b.ambang(nol.iloc[r.integers(0, len(nol), len(nol))], kol, 2)) for _ in range(1000)]
            print(f'    {j:8s} {s:4g} n={len(g):3d}: rerata {g[kol].mean():+.3f}, daya {b.laju(g, kol, lo, hi):.2f} '
                  f'[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]', flush=True)
    print('\n  rasio daya, AUC jendela dalam-pasien terhadap horizon (bootstrap berpasangan, ambang ikut diundi):')
    for hk in ('h_nsr', 'h_chf'):
        for j, s in SEL_W:
            g = h[(h.jenis == j) & (h.besar == s)].reset_index(drop=True)

            def dua(nb, gb):
                a = b.laju(gb, 'd_auc', *b.ambang(nb, 'd_auc', 2))
                c = b.laju(gb, hk, *b.ambang(nb, hk, 2))
                return a - c, a / c if c > 0 else np.nan
            s0, q0 = dua(nol, g)
            bs = np.array([dua(nol.iloc[r.integers(0, len(nol), len(nol))], g.iloc[r.integers(0, len(g), len(g))])
                           for _ in range(1000)])
            print(f'    {hk} {j:8s} {s:4g}: selisih {s0:+.2f} [{np.nanpercentile(bs[:, 0], 2.5):+.2f}, '
                  f'{np.nanpercentile(bs[:, 0], 97.5):+.2f}], rasio {q0:.1f} [{np.nanpercentile(bs[:, 1], 2.5):.1f}, '
                  f'{np.nanpercentile(bs[:, 1], 97.5):.1f}]', flush=True)


# ---------------------------------------------------------------- primer
def auc_ulang(d):
    X, y, g = d[CIRI].to_numpy(), d.y.to_numpy(), d.rec.to_numpy()
    ps = []
    for s in range(ULANG_P):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            p[te] = u.model('rf').set_params(n_jobs=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return roc_auc_score(y, np.mean(ps, axis=0))


def satu_primer(f0, arah, ids, jenis, besar, benih):
    """Satu kontras teramati (benih 7000 + k) atau satu undian pergeseran (benih 8000 + k) pada data bersuntik."""
    r = np.random.default_rng(benih)
    f = f0 if jenis == 'nol' else (y10.suntik_nadi(f0, besar, r) if jenis == 'nadi' else x.suntik(f0, besar, arah[jenis], r))
    if benih < 8000:
        d = x.dalam(f, None)
        d = d[d.rec.isin(ids)].reset_index(drop=True)
        peran = 'teramati'
    else:
        s = f[f.db == 'sddb']
        bag = []
        for rid in ids:
            g = s[s.rec == rid]
            m = r.uniform(120, g.menit_sebelum_vf.max() - 120)
            dk = g[(g.menit_sebelum_vf >= m) & (g.menit_sebelum_vf < m + 30)]
            jh = g[g.menit_sebelum_vf >= m + 120]
            if len(dk) < 3 or len(jh) == 0:
                continue
            jh = jh.sample(n=min(6, len(jh)), random_state=int(r.integers(1 << 30)))
            bag += [dk.assign(y=1), jh.assign(y=0)]
        d = pd.concat(bag).reset_index(drop=True)
        peran = 'geser'
    return dict(jenis=jenis, besar=besar, benih=benih, peran=peran, auc=auc_ulang(d), n_pasien=d.rec.nunique())


def primer():
    f0 = x.fitur()
    arah = arah_w(f0)
    s = f0[f0.db == 'sddb']
    utama = x.dalam(f0, None)
    panjang = s.groupby('rec').menit_sebelum_vf.max()
    ids = sorted(set(utama.rec) & set(panjang[panjang >= 240].index))
    print(f'\n== Daya bersyarat uji primer: {len(ids)} pasien, random forest, {ULANG_P} ulangan CV; tiap sel {N_P} '
          f'kontras teramati dan {N_P} undian pergeseran, suntikan sendiri-sendiri', flush=True)
    tugas = []
    for j, sb in SEL_P:
        n_obs = 1 if j == 'nol' else N_P           # tanpa suntikan kontras teramati deterministik
        tugas += [(j, sb, 7000 + k) for k in range(n_obs)] + [(j, sb, 8000 + k) for k in range(N_P)]
    path = os.path.join(OUT, 'd_primer.csv')
    b.jalankan(path, tugas, satu_primer, f0, arah, ids)
    ringkas_primer()


def ringkas_primer():
    h = pd.read_csv(os.path.join(OUT, 'd_primer.csv'))
    r = np.random.default_rng(41)
    print('\n== Ringkasan daya bersyarat uji primer (tolak bila teramati > persentil 95 pergeseran sel yang sama)')
    for (j, sb), g in h.groupby(['jenis', 'besar'], sort=False):
        ob, ge = g[g.peran == 'teramati'].auc.to_numpy(), g[g.peran == 'geser'].auc.to_numpy()
        q = np.percentile(ge, 95)
        bs = [(r.choice(ob, len(ob)) > np.percentile(r.choice(ge, len(ge)), 95)).mean() for _ in range(1000)]
        print(f'  {j:8s} {sb:4g}: teramati rerata {ob.mean():.3f} (n={len(ob)}), pergeseran rerata {ge.mean():.3f} '
              f'persentil 95 {q:.3f} (n={len(ge)}); daya {(ob > q).mean():.2f} [{np.percentile(bs, 2.5):.2f}, '
              f'{np.percentile(bs, 97.5):.2f}]; p rerata {np.mean([(1 + (ge >= o).sum()) / (len(ge) + 1) for o in ob]):.3f}',
              flush=True)


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['ragam', 'daya_w', 'primer']
    for bag in pilih:
        globals()[bag]()

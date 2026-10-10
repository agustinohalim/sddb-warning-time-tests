"""Artikel EKG 1 — analisis sesudah telaah panel keempat 10 Oktober 2026 (Telaah/2026-10-10-EKG1/report.md).

Masalah yang dijawab: pada 10_daya_ulang.py semua simulasi memakai 12 pasien yang sama, jadi simulasi
tanpa suntikan bukan ulangan nol yang bebas, dan aturan deteksi tidak terkalibrasi (0/100 di nol).
Mekanisme "ektopi membawa pemisahan" belum diuji, perubahan ektopi tidak disimulasikan dengan benar,
dan uji permutasi dalam-pasien bisa tertipu pergeseran waktu.

Bagian (argumen; tanpa argumen semua, urut):
  daya      simulasi daya dengan kohort bootstrap: tiap simulasi mengambil ulang pasien sddb (kohort
            tetap 12 untuk horizon, 19 untuk dalam-pasien) dan subjek nsrdb dengan pengembalian; duplikat
            tetap dalam satu lipatan (grup = rekaman asli). Regresi logistik, 100 simulasi per sel.
            Statistik: horizon = AUC(0) - AUC(6 jam), dihitung dua kali: model semua ciri dan model tanpa
            frac_ektopik; dalam-pasien = rerata selisih skor OOF dekat - jauh per pasien.
            Daya disesuaikan ukuran: ambang dari 100 simulasi tanpa suntikan (horizon dua sisi,
            persentil 2,5/97,5; dalam-pasien satu sisi, persentil 95); juga aturan nominal (bootstrap
            subjek dua sisi untuk horizon, Wilcoxon satu sisi untuk dalam-pasien) dengan laju positif
            palsunya di nol.
            Sel: tanpa suntikan; searah_k 1 dan 2 SD; tegak_k 1 SD; denyut +10 dan +20 per menit;
            ektopi naik x2 dan x4, turun x0,5 dan x0,25 (perkalian pada frac_ektopik, meluruh
            e^(-t/60), bobot per pasien; nilai nol tetap nol sehingga tidak ada artefak seri).
  blok      kontrol pergeseran waktu untuk kontras dalam-pasien pada data nyata: untuk tiap pasien,
            satu blok 30 menit berurutan yang berakhir >= 150 menit sebelum VF dijadikan "dekat palsu"
            dan dibandingkan dengan jendela jauh lain (>= 120 menit, di luar blok); 50 undian per model.
  permutasi uji permutasi dalam-pasien random forest dengan N = 999.
Keluaran: keluaran/daya_bootstrap.csv, keluaran/daya_bootstrap.log (lewat >)
Pakai:    python -u 11_daya_bootstrap.py [daya] [blok] [permutasi] > keluaran/daya_bootstrap.log
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
v = import_module('04_uji_lanjut')
w = import_module('07_dalam_pasien')
x = import_module('09_revisi_panel')
y10 = import_module('10_daya_ulang')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
TANPA_EKT = u.SET['tanpa_ektopik']
HOR = v.HOR_TETAP
ULANG_H, ULANG_D, BOOT = 5, 3, 300


# ---------------------------------------------------------------- suntikan
def kali_pasien(f, r):
    s = f.db == 'sddb'
    per = r.normal(1.0, 0.5, size=f.loc[s, 'rec'].nunique())
    return pd.Series(per, index=f.loc[s, 'rec'].unique()).reindex(f.loc[s, 'rec']).to_numpy()


def suntik_ektopi(f, faktor, r):
    """frac_ektopik dikali (1 + (faktor - 1) e^(-t/60) bobot); nol tetap nol."""
    f = f.copy()
    s = f.db == 'sddb'
    t = f.loc[s, 'menit_sebelum_vf'].to_numpy()
    k = kali_pasien(f, r)
    m = 1 + (faktor - 1) * np.exp(-t / 60.0) * k
    m = np.clip(m, 0.0, None)
    f.loc[s, 'frac_ektopik'] = np.clip(f.loc[s, 'frac_ektopik'].to_numpy() * m, 0.0, 1.0)
    return f


def buat_data(f0, arah, jenis, besar, r):
    if jenis == 'nol':
        return f0
    if jenis == 'nadi':
        return y10.suntik_nadi(f0, besar, r)
    if jenis == 'ektopi':
        return suntik_ektopi(f0, besar, r)
    return x.suntik(f0, besar, arah[jenis], r)


# ---------------------------------------------------------------- uji pada kohort bootstrap
def cv_skor(d, ciri, ulang):
    X, yy, g = d[ciri].to_numpy(), d.y.to_numpy(), d.grp.to_numpy()
    ps = []
    for b in range(ulang):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, yy, g):
            p[te] = u.model('logistik').fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def horizon(f, pas, kon, r):
    """AUC(0) - AUC(6 jam) untuk kohort bootstrap; dua set ciri. pas, kon: daftar id (boleh berulang)."""
    nsr = f[f.db == 'nsrdb']
    neg = []
    for k, rid in enumerate(kon):
        g = nsr[nsr.rec == rid]
        neg.append(g.sample(n=min(6, len(g)), random_state=int(r.integers(1 << 30)))
                   .assign(rec=f'n{rid}#{k}', grp=f'n{rid}'))
    neg = pd.concat(neg).assign(y=0)
    hasil = {}
    for nama, ciri in (('semua', CIRI), ('tanpa_ektopik', TANPA_EKT)):
        S, yy = [], None
        for h in HOR:
            pos = u.ambil_scd(f, h)
            bag = pd.concat([pos[pos.rec == rid].assign(rec=f's{rid}#{k}', grp=f's{rid}')
                             for k, rid in enumerate(pas)]).assign(y=1)
            d = pd.concat([bag, neg]).reset_index(drop=True)
            p = cv_skor(d, ciri, ULANG_H)
            sc = pd.Series(p).groupby(d.rec.to_numpy()).mean()
            if yy is None:
                yy = d.groupby('rec').y.first().reindex(sc.index)
            S.append(sc.reindex(yy.index).to_numpy())
        S = np.column_stack(S)
        yv = yy.to_numpy()
        sel = roc_auc_score(yv, S[:, 0]) - roc_auc_score(yv, S[:, -1])
        pi, ni = np.where(yv == 1)[0], np.where(yv == 0)[0]
        ds = []
        for _ in range(BOOT):
            ii = np.r_[r.choice(pi, len(pi)), r.choice(ni, len(ni))]
            ds.append(roc_auc_score(yv[ii], S[ii, 0]) - roc_auc_score(yv[ii], S[ii, -1]))
        ds = np.array(ds)
        pv = (ds < 0).mean() + 0.5 * (ds == 0).mean()
        hasil[nama] = (sel, bool(pv < 0.025 or pv > 0.975))
    return hasil


def dalam(f, pas, r):
    dd = x.dalam(f, r)
    d = pd.concat([dd[dd.rec == rid].assign(rec=f'{rid}#{k}', grp=rid) for k, rid in enumerate(pas)])
    d = d.reset_index(drop=True)
    p = cv_skor(d, CIRI, ULANG_D)
    t = d.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
    sel = (t[1] - t[0]).dropna()
    pw = wilcoxon(sel, alternative='greater').pvalue if (sel != 0).any() else 1.0
    return sel.mean(), bool(pw < 0.025)


def daya():
    f0 = x.fitur()
    arah = y10.arah_dasar(f0)
    tetap = x.kohort_tetap(f0)
    dal_ids = sorted(x.dalam(f0, np.random.default_rng(0)).rec.unique())
    kon_ids = sorted(f0[f0.db == 'nsrdb'].rec.unique())
    sel = [('nol', 0)] + [('searah_k', 1.0), ('searah_k', 2.0), ('tegak_k', 1.0), ('nadi', 10), ('nadi', 20),
                          ('ektopi', 2.0), ('ektopi', 4.0), ('ektopi', 0.5), ('ektopi', 0.25)]
    path = os.path.join(OUT, 'daya_bootstrap.csv')
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    print(f'== Simulasi daya kohort bootstrap: horizon {len(tetap)} pasien, dalam-pasien {len(dal_ids)}, '
          f'nsrdb {len(kon_ids)}', flush=True)
    for jenis, besar in sel:
        t0 = time.time()
        for k in range(100):
            if len(sudah) and ((sudah.jenis == jenis) & (sudah.besar == besar) & (sudah.sim == k)).any():
                continue
            r = np.random.default_rng(5000 + k)
            pas = list(r.choice(tetap, len(tetap)))
            kon = list(r.choice(kon_ids, len(kon_ids)))
            pd_ = list(r.choice(dal_ids, len(dal_ids)))
            f = buat_data(f0, arah, jenis, besar, r)
            h = horizon(f, pas, kon, r)
            ds, dok = dalam(f, pd_, r)
            baris = dict(jenis=jenis, besar=besar, sim=k, h_sel=h['semua'][0], h_nom=h['semua'][1],
                         h_sel_te=h['tanpa_ektopik'][0], h_nom_te=h['tanpa_ektopik'][1], d_sel=ds, d_nom=dok)
            pd.DataFrame([baris]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        print(f'  {jenis:8s} {besar:5g}: {time.time() - t0:.0f} dtk', flush=True)
    ringkas()


def ringkas():
    h = pd.read_csv(os.path.join(OUT, 'daya_bootstrap.csv'))
    n = h[h.jenis == 'nol']
    lo, hi = n.h_sel.quantile([0.025, 0.975])
    lo2, hi2 = n.h_sel_te.quantile([0.025, 0.975])
    dt = n.d_sel.quantile(0.95)
    print(f'\n== Ringkasan (ambang dari {len(n)} simulasi nol: horizon [{lo:+.3f}, {hi:+.3f}], tanpa ektopi '
          f'[{lo2:+.3f}, {hi2:+.3f}], dalam-pasien > {dt:+.3f})')
    print(f'  nominal di nol: horizon {n.h_nom.mean():.2f}, horizon tanpa ektopi {n.h_nom_te.mean():.2f}, '
          f'dalam-pasien {n.d_nom.mean():.2f}')
    for (j, b), g in h.groupby(['jenis', 'besar'], sort=False):
        a1 = ((g.h_sel < lo) | (g.h_sel > hi)).mean()
        a2 = ((g.h_sel_te < lo2) | (g.h_sel_te > hi2)).mean()
        a3 = (g.d_sel > dt).mean()
        print(f'  {j:8s} {b:5g} n={len(g):3d}: horizon sel {g.h_sel.mean():+.3f} (sd {g.h_sel.std():.3f}) '
              f'disesuaikan {a1:.2f} nominal {g.h_nom.mean():.2f} | tanpa ektopi sel {g.h_sel_te.mean():+.3f} '
              f'disesuaikan {a2:.2f} nominal {g.h_nom_te.mean():.2f} | dalam-pasien sel {g.d_sel.mean():+.3f} '
              f'disesuaikan {a3:.2f} nominal {g.d_nom.mean():.2f}', flush=True)


# ---------------------------------------------------------------- blok (kontrol pergeseran waktu)
def blok():
    f = x.fitur()
    s = f[f.db == 'sddb']
    dekat = x.dalam(f, np.random.default_rng(5))
    ids = sorted(dekat.rec.unique())
    print('\n== Kontrol pergeseran waktu: blok 30 menit berakhir >= 150 menit sebelum VF sebagai "dekat palsu"',
          flush=True)
    for jenis, ulang in (('logistik', 3), ('rf', 2)):
        w.ULANG = ulang
        r = np.random.default_rng(11)
        a, sel_med = [], []
        t0 = time.time()
        for k in range(50):
            bag = []
            for rid in ids:
                g = s[(s.rec == rid) & (s.menit_sebelum_vf >= 150)].sort_values('menit_sebelum_vf')
                if len(g) < 12:
                    continue
                i = r.integers(0, len(g) - 6)
                blk = g.iloc[i:i + 6]
                # blok berurutan: 6 jendela 5 menit yang bersambung
                if blk.menit_sebelum_vf.max() - blk.menit_sebelum_vf.min() > 30:
                    continue
                sisa = s[(s.rec == rid) & (s.menit_sebelum_vf >= 120) & ~s.index.isin(blk.index)]
                sisa = sisa[(sisa.menit_sebelum_vf < blk.menit_sebelum_vf.min() - 30)
                            | (sisa.menit_sebelum_vf > blk.menit_sebelum_vf.max() + 30)]
                sisa = sisa.sample(n=min(6, len(sisa)), random_state=int(r.integers(1 << 30)))
                if len(sisa) == 0:
                    continue
                bag.append(blk.assign(y=1))
                bag.append(sisa.assign(y=0))
            d = pd.concat(bag).reset_index(drop=True)
            p = w.oof(d, jenis, CIRI)
            a.append(roc_auc_score(d.y, p))
            t = d.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
            sel_med.append((t[1] - t[0]).dropna().median())
        a = np.array(a)
        print(f'  {jenis:8s} AUC blok palsu rerata {a.mean():.3f} [{np.percentile(a, 2.5):.3f}, '
              f'{np.percentile(a, 97.5):.3f}], maks {a.max():.3f}; pasien per undian ~{d.rec.nunique()}; '
              f'median selisih skor {np.mean(sel_med):+.3f}; {time.time() - t0:.0f} dtk', flush=True)


# ---------------------------------------------------------------- permutasi 999 (RF)
def permutasi():
    f = x.fitur()
    d = x.dalam(f, np.random.default_rng(5))
    w.ULANG = 2
    p0 = w.oof(d, 'rf', CIRI)
    a0 = roc_auc_score(d.y, p0)
    r = np.random.default_rng(17)
    ap = []
    t0 = time.time()
    N = 999
    for k in range(N):
        dp = d.copy()
        dp['y'] = dp.groupby('rec').y.transform(lambda s_: r.permutation(s_.to_numpy()))
        ap.append(roc_auc_score(dp.y, w.oof(dp, 'rf', CIRI)))
        if (k + 1) % 100 == 0:
            print(f'  ... {k + 1} permutasi, {time.time() - t0:.0f} dtk', flush=True)
    ap = np.array(ap)
    p = (1 + (ap >= a0).sum()) / (N + 1)
    print(f'\n== Permutasi dalam-pasien RF N = {N}: AUC asli {a0:.3f}; permutasi rerata {ap.mean():.3f} '
          f'[{np.percentile(ap, 2.5):.3f}, {np.percentile(ap, 97.5):.3f}]; p {p:.4f}', flush=True)


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['daya', 'blok', 'permutasi']
    for b in pilih:
        globals()[b]()

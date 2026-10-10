"""Artikel EKG 1 — analisis sesudah telaah panel ketiga 9 Oktober 2026 (Telaah/2026-10-09-EKG1/report.md).

Diagnosis (Hasil_Uji_1.md): pada simulasi daya 09_revisi_panel.py batas bawah selang bootstrap uji
horizon menempel tepat di 0 pada langit-langit AUC, sehingga aturan "persentil 2,5 > 0" tidak pernah
terpenuhi. Skrip ini mengulang simulasi dengan aturan yang menangani seri dan dengan skenario yang lebih
adil, menambah uji permutasi dalam-pasien pada data nyata, dan memeriksa hitungan AF sddb.

Bagian (argumen; tanpa argumen semua, urut):
  af        per rekaman sddb: daftar tanda irama di aux_note, lama AF/flutter (menit) sepanjang rekaman
            dan di jendela pra-VF, dibandingkan dengan 4 pasien AF menurut dokumentasi PhysioNet
  permutasi kontras dalam-pasien pada data nyata (dekat 0-30 menit lawan >= 2 jam): label dekat/jauh
            diacak DI DALAM tiap pasien, model dilatih ulang tiap permutasi; p = (1 + #perm >= asli)/(N + 1);
            logistik N = 199, random forest N = 99
  daya      simulasi daya ulang. Arah perubahan pra-VF buatan (meluruh e^(-t/60 menit), bobot per pasien
            N(1; 0,5)):
              searah   searah perbedaan sddb - nsrdb (seperti 09), delta 0 / 0,5 / 1 / 2 SD
              balik    berlawanan arah, delta sama
              tegak    satu arah acak tegak lurus terhadap arah sddb - nsrdb (benih tetap), delta sama
              nadi     denyut jantung naik X denyut/menit (hanya mean_nn), X = 0 / 5 / 10 / 20
            Per simulasi:
              uji horizon kohort tetap lawan nsrdb, AUC(0) - AUC(360), 500 resampel subjek tanpa latih
                ulang; terdeteksi bila p satu sisi < 0,025 dengan p = P(selisih < 0) + 0,5 P(selisih = 0)
                (seri di 0 dihitung setengah); dicatat juga aturan lama (persentil 2,5 > 0) dan lebar selang
              uji dalam-pasien (dekat 0-30 menit lawan >= 2 jam, pasien yang sama): selisih rerata skor OOF
                dekat - jauh per pasien, Wilcoxon satu sisi p < 0,025 (tingkat pasien, tanpa bootstrap)
            Logistik 100 simulasi tanpa suntikan dan 50 per sel lain; random forest 10 per sel (searah 0 / 1 / 2 SD,
            nadi 20), karena ~6 menit per simulasi. Sel besar 0 dijalankan sekali karena identik untuk semua arah.
Keluaran: keluaran/daya_ulang.csv (satu baris per simulasi), keluaran/daya_ulang.log (lewat >)
Pakai:    python -u 10_daya_ulang.py [af] [permutasi] [daya] > keluaran/daya_ulang.log
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
v = import_module('04_uji_lanjut')
w = import_module('07_dalam_pasien')
x = import_module('09_revisi_panel')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
THESIS = x.THESIS


# ---------------------------------------------------------------- af
def af():
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    ir = pd.read_csv(os.path.join(OUT, 'irama_jendela.csv'), dtype={'rec': str})
    f = pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str})
    print('== AF/flutter per rekaman sddb (aux_note .ari); PhysioNet: 4 pasien AF')
    for _, r in rek[rek.db == 'sddb'].iterrows():
        a = wfdb.rdann(os.path.join(THESIS, 'sddb', r.rec), 'ari')
        t = np.asarray(a.sample) / r.fs
        aux = [s.strip('\x00') for s in a.aux_note]
        tanda = [(ti, s) for ti, s in zip(t, aux) if s.startswith('(')]
        akhir = t[-1] if len(t) else 0
        lama = {}
        for i, (ti, s) in enumerate(tanda):
            tj = tanda[i + 1][0] if i + 1 < len(tanda) else akhir
            lama[s] = lama.get(s, 0) + (tj - ti) / 60
        afm = sum(m for s, m in lama.items() if s.startswith(('(AFIB', '(AFL', '(AF')))
        g = ir[(ir.db == 'sddb') & (ir.rec == r.rec)]
        h = f[(f.db == 'sddb') & (f.rec == r.rec)]
        prevf = g.merge(h[['rec', 'mulai_s', 'menit_sebelum_vf']], on=['rec', 'mulai_s'])
        prevf = prevf[prevf.menit_sebelum_vf.notna()]
        ringkas = ', '.join(f'{s} {m:.0f}' for s, m in sorted(lama.items(), key=lambda z: -z[1])[:6])
        print(f'  {r.rec:>3s}  AF/flutter {afm:7.1f} menit dari {akhir / 60:6.0f}; jendela pra-VF AF '
              f'{int(prevf.af.sum()):4d}/{len(prevf):4d}  | {ringkas}', flush=True)


# ---------------------------------------------------------------- permutasi
def permutasi():
    f = x.fitur()
    d = x.dalam(f, np.random.default_rng(5))
    print('\n== Uji permutasi dalam-pasien (dekat 0-30 menit lawan >= 2 jam), label diacak di dalam pasien', flush=True)
    for jenis, N, ulang in (('logistik', 199, 3), ('rf', 99, 2)):
        w.ULANG = ulang
        p0 = w.oof(d, jenis, CIRI)
        a0 = roc_auc_score(d.y, p0)
        r = np.random.default_rng(7)
        ap = []
        t0 = time.time()
        for k in range(N):
            dp = d.copy()
            dp['y'] = dp.groupby('rec').y.transform(lambda s: r.permutation(s.to_numpy()))
            ap.append(roc_auc_score(dp.y, w.oof(dp, jenis, CIRI)))
        ap = np.array(ap)
        p = (1 + (ap >= a0).sum()) / (N + 1)
        print(f'  {jenis:8s} AUC asli {a0:.3f} (ulangan CV {ulang}); permutasi rerata {ap.mean():.3f} '
              f'[{np.percentile(ap, 2.5):.3f}, {np.percentile(ap, 97.5):.3f}]; p {p:.3f} (N={N}); '
              f'{time.time() - t0:.0f} dtk', flush=True)


# ---------------------------------------------------------------- daya
def arah_dasar(f0):
    sd = f0[CIRI].std().to_numpy()
    beda = (f0[f0.db == 'sddb'][CIRI].mean() - f0[f0.db == 'nsrdb'][CIRI].mean()).to_numpy()
    e = beda / sd
    e = e / np.linalg.norm(e)
    z = np.random.default_rng(42).normal(size=len(CIRI))
    z = z - (z @ e) * e
    z = z / np.linalg.norm(z)
    arah = {'searah': e * sd, 'balik': -e * sd, 'tegak': z * sd}
    # varian _k: hanya ciri kontinu. frac_ektopik bernilai tepat 0 pada 97,7 % jendela nsrdb dan 34 % sddb
    # (tidak dilog) dan pnn50 = 0 pada 13 % sddb; geseran sekecil apa pun memisahkan nilai seri itu,
    # dan random forest lalu memisahkan sddb dari nsrdb di horizon mana pun (artefak, lihat Hasil_Uji_1.md)
    kont = np.array([c not in ('frac_ektopik', 'pnn50') for c in CIRI], float)
    ek = e * kont
    ek = ek / np.linalg.norm(ek)
    zk = np.random.default_rng(42).normal(size=len(CIRI)) * kont
    zk = zk - (zk @ ek) * ek
    zk = zk / np.linalg.norm(zk)
    arah.update({'searah_k': ek * sd, 'balik_k': -ek * sd, 'tegak_k': zk * sd})
    return arah


def suntik_nadi(f, bpm, r):
    f = f.copy()
    s = f.db == 'sddb'
    t = f.loc[s, 'menit_sebelum_vf'].to_numpy()
    per = r.normal(1.0, 0.5, size=f.loc[s, 'rec'].nunique())
    kali = pd.Series(per, index=f.loc[s, 'rec'].unique()).reindex(f.loc[s, 'rec']).to_numpy()
    tambah = bpm * np.exp(-t / 60.0) * kali
    nn = f.loc[s, 'mean_nn'].to_numpy()
    f.loc[s, 'mean_nn'] = 60.0 / (60.0 / nn + tambah)
    return f


def satu_simulasi(f, r, jenis):
    nsr, _ = x.kontrol(f, r)
    tetap = x.kohort_tetap(f)
    a = []
    for h in v.HOR_TETAP:
        pos = u.ambil_scd(f, h)
        v.ULANG = 10 if jenis == 'logistik' else 2
        sc, subj = v.skor_subjek(pos[pos.rec.isin(tetap)], nsr, 'semua', jenis)
        a.append((subj, sc))
    idx = a[0][1].index
    y = a[0][0].reindex(idx).to_numpy()
    S = np.column_stack([z[1].reindex(idx).to_numpy() for z in a])
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    ds = []
    for _ in range(500):
        ii = np.r_[r.choice(pi, len(pi)), r.choice(ni, len(ni))]
        ds.append(roc_auc_score(y[ii], S[ii, 0]) - roc_auc_score(y[ii], S[ii, -1]))
    ds = np.array(ds)
    d0 = roc_auc_score(y, S[:, 0]) - roc_auc_score(y, S[:, -1])
    p_hor = (ds < 0).mean() + 0.5 * (ds == 0).mean()
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # dalam-pasien, tingkat pasien
    dd = x.dalam(f, r)
    w.ULANG = 3 if jenis == 'logistik' else 2
    p = w.oof(dd, jenis, CIRI)
    t = dd.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
    sel = (t[1] - t[0]).dropna()
    p_dal = wilcoxon(sel, alternative='greater').pvalue if (sel != 0).any() else 1.0
    return dict(auc0=roc_auc_score(y, S[:, 0]), auc6=roc_auc_score(y, S[:, -1]), selisih=d0, lo=lo, hi=hi,
                p_horizon=p_hor, horizon_lama=bool(lo > 0), horizon_baru=bool(p_hor < 0.025),
                dalam_median=sel.median(), dalam_n=len(sel), p_dalam=p_dal, dalam_ok=bool(p_dal < 0.025))


def daya():
    f0 = x.fitur()
    arah = arah_dasar(f0)
    jalan = []
    # besar 0 identik untuk semua arah (benih sama, tanpa suntikan), jadi hanya dijalankan sekali (searah)
    jalan += [('logistik', 'searah', 0.0, 100)]
    for nama in ('searah', 'balik', 'tegak'):
        jalan += [('logistik', nama, dl, 50) for dl in (0.5, 1.0, 2.0)]
    jalan += [('logistik', 'nadi', b, 50) for b in (5, 10, 20)]
    # random forest ~6 menit per simulasi: 10 simulasi, sel terpilih saja
    jalan += [('rf', 'searah', dl, 10) for dl in (0.0, 1.0, 2.0)]
    jalan += [('rf', 'nadi', 20, 10)]
    # ulangan dengan suntikan pada ciri kontinu saja (tanpa artefak seri)
    for nama in ('searah_k', 'balik_k', 'tegak_k'):
        jalan += [('logistik', nama, dl, 50) for dl in (0.5, 1.0, 2.0)]
    for nama in ('searah_k', 'balik_k', 'tegak_k'):
        jalan += [('rf', nama, dl, 10) for dl in (1.0, 2.0)]
    path = os.path.join(OUT, 'daya_ulang.csv')
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    print('\n== Simulasi daya ulang', flush=True)
    for jenis, nama, besar, nsim in jalan:
        t0 = time.time()
        for k in range(nsim):
            if len(sudah) and ((sudah.model == jenis) & (sudah.arah == nama) & (sudah.besar == besar)
                               & (sudah.sim == k)).any():
                continue
            r = np.random.default_rng(1000 + k)
            f = suntik_nadi(f0, besar, r) if nama == 'nadi' else x.suntik(f0, besar, arah[nama], r)
            hasil = dict(model=jenis, arah=nama, besar=besar, sim=k, **satu_simulasi(f, r, jenis))
            pd.DataFrame([hasil]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        h = pd.read_csv(path)
        h = h[(h.model == jenis) & (h.arah == nama) & (h.besar == besar)]
        print(f'  {jenis:8s} {nama:8s} {besar:4g}: selisih rerata {h.selisih.mean():+.3f}, lebar selang rerata '
              f'{(h.hi - h.lo).mean():.3f}; horizon terdeteksi {h.horizon_baru.sum()}/{len(h)} '
              f'(aturan lama {h.horizon_lama.sum()}); dalam-pasien {h.dalam_ok.sum()}/{len(h)} '
              f'(median selisih skor {h.dalam_median.mean():+.3f}); {time.time() - t0:.0f} dtk', flush=True)


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['af', 'permutasi', 'daya']
    for b in pilih:
        globals()[b]()

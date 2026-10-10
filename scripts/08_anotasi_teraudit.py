"""Artikel EKG 1 — apakah hasil bergantung pada anotasi .ari yang belum diaudit?

Menjawab isu telaah panel (28 September 2026, disepakati tiga peran): fraksi denyut ektopik,
ciri terkuat, dihitung dari anotasi .ari otomatis untuk sddb tetapi dari .atr teraudit untuk
nsrdb. sddb punya .atr teraudit penuh untuk 12 rekaman; 11 di antaranya punya awitan VF.

Pada 11 pasien yang SAMA, seluruh analisis dijalankan dua kali — ciri dari .ari dan ciri dari
.atr — dengan fungsi yang sama persis dengan 02_fitur.py, 03_uji.py dan 07_dalam_pasien.py.
Bila hasilnya nyaris sama, sumber anotasi tidak mendorong pemisahan.

  A  kesepakatan per jendela antara ciri .ari dan .atr (Spearman, selisih median)
  H  uji horizon lawan nsrdb dan chfdb, 0/1/3/6 jam, set ciri semua dan ektopik saja
  W  kontras dalam-pasien (0-30 menit lawan >= 2, 3, 6 jam sebelumnya)

Kontrol nsrdb dan chfdb memakai jendela dari keluaran/fitur.csv dengan benih yang sama untuk
kedua sumber, jadi yang berbeda hanya anotasi sddb.

Keluaran: keluaran/kesepakatan_anotasi.csv, keluaran/hasil_teraudit.csv,
          keluaran/hasil_teraudit_dalam_pasien.csv
Pakai:    python 08_anotasi_teraudit.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
SDDB = os.path.join(os.path.join(HERE, '..', 'data'), 'sddb')


def muat(nama):
    spec = importlib.util.spec_from_file_location(nama.replace('.py', ''), os.path.join(HERE, nama))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FITUR, UJI, DALAM = muat('02_fitur.py'), muat('03_uji.py'), muat('07_dalam_pasien.py')
SET_UJI = ['semua', 'ektopik_saja']
HORIZON = [0, 60, 180, 360]      # subset horizon 03_uji; 12 jam hanya 6 pasien teraudit


def jendela_sddb(rec, vf_s, ext):
    """Ciri per jendela 5 menit sebelum VF, persis seperti 02_fitur.main untuk sddb."""
    p = os.path.join(SDDB, rec)
    fs = wfdb.rdheader(p).fs
    a = wfdb.rdann(p, ext)
    t = np.asarray(a.sample) / fs
    sym = np.asarray(a.symbol).astype(str)
    baris = []
    for s in np.arange(0, vf_s - FITUR.JENDELA + 1, FITUR.JENDELA):
        m = (t >= s) & (t < s + FITUR.JENDELA)
        c = FITUR.ciri(t[m], sym[m])
        if c is None:
            continue
        c.update({'db': 'sddb', 'rec': rec, 'mulai_s': s,
                  'menit_sebelum_vf': (vf_s - (s + FITUR.JENDELA)) / 60})
        baris.append(c)
    return pd.DataFrame(baris)


def kesepakatan(ari, atr):
    g = ari.merge(atr, on=['rec', 'mulai_s'], suffixes=('_ari', '_atr'))
    baris = []
    for c in FITUR.__dict__.get('CIRI', None) or UJI.CIRI_SEMUA:
        x, y = g[f'{c}_ari'], g[f'{c}_atr']
        ok = x.notna() & y.notna()
        rho = spearmanr(x[ok], y[ok]).correlation
        baris.append({'ciri': c, 'jendela_bersama': int(ok.sum()), 'spearman': rho,
                      'median_ari': x[ok].median(), 'median_atr': y[ok].median(),
                      'median_selisih': (x[ok] - y[ok]).median()})
    per_rek = g.groupby('rec')[['frac_ektopik_ari', 'frac_ektopik_atr']].mean()
    return pd.DataFrame(baris), per_rek, len(ari), len(atr), len(g)


def main():
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    rek = rek[(rek.db == 'sddb') & rek.vf_s.notna()]
    audit = [r for r in rek.rec if os.path.exists(os.path.join(SDDB, f'{r}.atr'))]
    vf = dict(zip(rek.rec, rek.vf_s))
    print(f'rekaman sddb dengan .atr teraudit dan awitan VF: {len(audit)} -> {", ".join(audit)}\n')

    ari = pd.concat([jendela_sddb(r, vf[r], 'ari') for r in audit], ignore_index=True)
    atr = pd.concat([jendela_sddb(r, vf[r], 'atr') for r in audit], ignore_index=True)

    # A. kesepakatan
    k, per_rek, n_ari, n_atr, n_sama = kesepakatan(ari, atr)
    k.to_csv(os.path.join(OUT, 'kesepakatan_anotasi.csv'), index=False)
    print(f'A  jendela: .ari {n_ari}, .atr {n_atr}, bersama {n_sama}')
    for _, r in k.iterrows():
        print(f'   {r.ciri:13s} Spearman {r.spearman:.3f}  median ari {r.median_ari:.4g} atr {r.median_atr:.4g}'
              f'  selisih median {r.median_selisih:+.4g}')
    print('   fraksi ektopik per rekaman (rerata jendela), ari / atr:')
    for rec, r in per_rek.iterrows():
        print(f'     {rec}: {r.frac_ektopik_ari:.3f} / {r.frac_ektopik_atr:.3f}')

    kontrol = pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str})
    kontrol = kontrol[kontrol.db != 'sddb']

    # H. uji horizon, sumber anotasi diganti, pasien sama
    baris = []
    for sumber, sd in (('ari', ari), ('atr', atr)):
        f = UJI.siapkan(pd.concat([kontrol, sd], ignore_index=True))
        rng = np.random.default_rng(2026)          # kontrol identik untuk kedua sumber
        nsr = UJI.ambil_kontrol(f, 'nsrdb', rng)
        chf = UJI.ambil_kontrol(f, 'chfdb', rng)
        for jenis in ('logistik', 'rf'):
            for cs in SET_UJI:
                for h in HORIZON:
                    scd = UJI.ambil_scd(f, h)
                    if scd.rec.nunique() < 5:
                        continue
                    for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
                        r = UJI.uji(f, scd, neg, f'{nama}_{h}', cs, jenis, rng)
                        r.update(sumber=sumber, kontrol=nama, horizon_menit=h)
                        baris.append(r)
                        print(f'H  {sumber} {jenis:8s} {cs:13s} {nama} h={h:4d} AUC {r["auc"]:.3f} '
                              f'[{r["lo"]:.2f},{r["hi"]:.2f}] n={r["n_pos"]}/{r["n_neg"]}', flush=True)
                        pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_teraudit.csv'), index=False)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_teraudit.csv'), index=False)

    # W. dalam-pasien, sumber anotasi diganti
    baris = []
    for sumber, sd in (('ari', ari), ('atr', atr)):
        f = sd.copy()
        for c in DALAM.LOG:
            f[c] = np.log(f[c].clip(lower=1e-8))
        f['lf_hf'] = f['lf_hf'].fillna(f['lf_hf'].median())
        rng = np.random.default_rng(2026)
        pos = f[f.menit_sebelum_vf < 30]
        n = pos.groupby('rec').size()
        pos = pos[pos.rec.isin(n[n >= 3].index)]
        for jauh in DALAM.JAUH:
            neg = f[(f.menit_sebelum_vf >= jauh) & f.rec.isin(pos.rec)]
            neg = neg.sample(frac=1, random_state=int(rng.integers(1 << 30))).groupby('rec').head(DALAM.PER)
            if neg.rec.nunique() < 5:
                continue
            d = pd.concat([pos[pos.rec.isin(neg.rec)].assign(y=1), neg.assign(y=0)]).reset_index(drop=True)
            for j in ('logistik', 'rf'):
                for nama, ciri in (('semua', DALAM.CIRI), ('tanpa_ektopik', DALAM.CIRI[:-1])):
                    a, lo, hi = DALAM.boot(d, DALAM.oof(d, j, ciri), rng)
                    baris.append(dict(sumber=sumber, jauh=jauh, model=j, ciri=nama, auc=a, lo=lo, hi=hi,
                                      n_pasien=d.rec.nunique()))
                    print(f'W  {sumber} jauh>={jauh:3d} {j:8s} {nama:13s} AUC {a:.3f} [{lo:.2f},{hi:.2f}] '
                          f'pasien={d.rec.nunique()}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_teraudit_dalam_pasien.csv'), index=False)


if __name__ == '__main__':
    main()

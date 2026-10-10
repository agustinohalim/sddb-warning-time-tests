"""Artikel EKG 1 — analisis sesudah telaah panel keenam (Telaah/2026-10-11b-EKG1/report.md).

Bagian:
  lambat    simulasi daya dengan perubahan pra-VF yang meluruh lebih lambat (e^(-t/180 menit)), sehingga
            jendela jauh (>= 2 jam) dan horizon 6 jam ikut berubah. Sel: searah_k 1 dan 2 SD, denyut +20;
            100 simulasi, benih dan pengambilan ulang pasien sama dengan 11/12 (12 pasien kohort tetap
            untuk kedua uji). Ambang nol dipakai ulang dari simulasi nol yang sudah ada (tidak bergantung
            pada laju peluruhan): horizon dari daya_bootstrap.csv, dalam-pasien dari akhir_dalam12.csv.
  rasio     selisih dan rasio daya dalam-pasien terhadap horizon dengan selang bootstrap berpasangan atas
            simulasi (kedua uji memakai pengambilan ulang pasien yang sama per simulasi).
Keluaran: keluaran/lambat.csv, keluaran/lambat.log (lewat >)
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
x = import_module('09_revisi_panel')
y10 = import_module('10_daya_ulang')
b11 = import_module('11_daya_bootstrap')
c12 = import_module('12_analisis_akhir')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
TAU = 180.0


def suntik_lambat(f, delta, arah, r):
    f = f.copy()
    s = f.db == 'sddb'
    t = f.loc[s, 'menit_sebelum_vf'].to_numpy()
    k = b11.kali_pasien(f, r)
    f.loc[s, CIRI] = f.loc[s, CIRI].to_numpy() + np.outer(delta * np.exp(-t / TAU) * k, arah)
    return f


def nadi_lambat(f, bpm, r):
    f = f.copy()
    s = f.db == 'sddb'
    t = f.loc[s, 'menit_sebelum_vf'].to_numpy()
    k = b11.kali_pasien(f, r)
    nn = f.loc[s, 'mean_nn'].to_numpy()
    f.loc[s, 'mean_nn'] = 60.0 / (60.0 / nn + bpm * np.exp(-t / TAU) * k)
    return f


def lambat():
    f0 = x.fitur()
    arah = y10.arah_dasar(f0)
    tetap = x.kohort_tetap(f0)
    kon_ids = sorted(f0[f0.db == 'nsrdb'].rec.unique())
    path = os.path.join(OUT, 'lambat.csv')
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    print(f'== Peluruhan lambat (tau {TAU:g} menit), 12 pasien kohort tetap, kedua uji', flush=True)
    for jenis, besar in (('searah_k', 1.0), ('searah_k', 2.0), ('nadi', 20)):
        t0 = time.time()
        for k in range(100):
            if len(sudah) and ((sudah.jenis == jenis) & (sudah.besar == besar) & (sudah.sim == k)).any():
                continue
            r = np.random.default_rng(5000 + k)
            pas = list(r.choice(tetap, len(tetap)))
            kon = list(r.choice(kon_ids, len(kon_ids)))
            f = nadi_lambat(f0, besar, r) if jenis == 'nadi' else suntik_lambat(f0, besar, arah[jenis], r)
            hs = c12.horizon_stat(f, pas, kon, r, 'logistik', b11.ULANG_H)
            ds = c12.dalam_stat(f, pas, r)
            pd.DataFrame([dict(jenis=jenis, besar=besar, sim=k, h=hs, d=ds)]).to_csv(
                path, mode='a', header=not os.path.exists(path), index=False)
        print(f'  {jenis:8s} {besar:5g}: {time.time() - t0:.0f} dtk', flush=True)
    hb = pd.read_csv(os.path.join(OUT, 'daya_bootstrap.csv'))
    d12 = pd.read_csv(os.path.join(OUT, 'akhir_dalam12.csv'))
    lo_h, hi_h = hb[hb.jenis == 'nol'].h_sel.quantile([0.025, 0.975])
    lo_d, hi_d = d12[d12.jenis == 'nol'].d12.quantile([0.025, 0.975])
    L = pd.read_csv(path)
    for (j, b), g in L.groupby(['jenis', 'besar'], sort=False):
        ph = ((g.h < lo_h) | (g.h > hi_h)).mean()
        pdl = ((g.d < lo_d) | (g.d > hi_d)).mean()
        print(f'  {j:8s} {b:5g}: horizon selisih {g.h.mean():+.3f} daya {ph:.2f} | dalam-pasien selisih '
              f'{g.d.mean():+.3f} daya {pdl:.2f}', flush=True)


def rasio():
    hb = pd.read_csv(os.path.join(OUT, 'daya_bootstrap.csv'))
    d12 = pd.read_csv(os.path.join(OUT, 'akhir_dalam12.csv'))
    lo_h, hi_h = hb[hb.jenis == 'nol'].h_sel.quantile([0.025, 0.975])
    lo_d, hi_d = d12[d12.jenis == 'nol'].d12.quantile([0.025, 0.975])
    m = hb.merge(d12, on=['jenis', 'besar', 'sim'])
    m['dh'] = (m.h_sel < lo_h) | (m.h_sel > hi_h)
    m['dd'] = (m.d12 < lo_d) | (m.d12 > hi_d)
    r = np.random.default_rng(3)
    print('\n== Selisih dan rasio daya dalam-pasien (12) terhadap horizon (12), bootstrap berpasangan atas simulasi')
    for (j, b), g in m.groupby(['jenis', 'besar'], sort=False):
        if j == 'nol':
            continue
        h, d = g.dh.to_numpy(), g.dd.to_numpy()
        sel, rs = [], []
        for _ in range(2000):
            i = r.integers(0, len(g), len(g))
            sel.append(d[i].mean() - h[i].mean())
            rs.append(d[i].mean() / max(h[i].mean(), 0.01))
        print(f'  {j:8s} {b:5g}: horizon {h.mean():.2f} dalam {d.mean():.2f} | selisih {d.mean() - h.mean():+.2f} '
              f'[{np.percentile(sel, 2.5):+.2f}, {np.percentile(sel, 97.5):+.2f}] | rasio '
              f'{d.mean() / max(h.mean(), 0.01):.1f} [{np.percentile(rs, 2.5):.1f}, {np.percentile(rs, 97.5):.1f}]',
              flush=True)


if __name__ == '__main__':
    for b in sys.argv[1:] or ['rasio', 'lambat']:
        globals()[b]()

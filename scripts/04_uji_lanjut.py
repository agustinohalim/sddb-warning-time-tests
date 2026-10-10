"""Artikel EKG 1 — uji awitan palsu dan uji kecenderungan horizon pada kohort tetap.

A  Awitan palsu. Untuk setiap rekaman sddb, awitan VF dipindah ke titik acak paling lambat 60
   menit sebelum awitan sebenarnya. Jendela SCD diambil 0-30 menit sebelum awitan palsu, lalu
   dibandingkan dengan nsrdb dan chfdb seperti uji 1. Diulang 20 undian. Bila AUC awitan palsu
   setara AUC awitan sebenarnya, label waktunya tidak membawa informasi.

T  Kecenderungan horizon pada KOHORT TETAP: hanya subjek sddb yang punya jendela di semua
   horizon 0-360 menit (uji 1 memakai subjek yang berbeda-beda per horizon). Kemiringan AUC
   terhadap horizon (per jam) dan selisih AUC(0) - AUC(360) dengan selang bootstrap subjek —
   undian subjek yang sama dipakai untuk semua horizon, jadi selisihnya berpasangan.

Pembagian, model, dan ciri sama dengan 03_uji.py.

Keluaran: keluaran/hasil_uji_lanjut.csv dan ringkasan di layar.
Pakai:    python 04_uji_lanjut.py
"""
import os
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

u = import_module('03_uji')
OUT = u.OUT
ULANG = 10
UNDIAN_PALSU = 20
HOR_TETAP = [0, 30, 60, 120, 180, 360]


def skor_subjek(pos, neg, ciri, jenis):
    d = pd.concat([pos.assign(y=1), neg.assign(y=0)])
    d = d.assign(rec=d.db + ':' + d.rec)
    ss = []
    for b in range(ULANG):
        s, subj, _ = u.skor_cv(d, u.SET[ciri], jenis, b)
        ss.append(s)
    return pd.concat(ss, axis=1).mean(axis=1), subj


def uji_palsu(f, nsr, chf, rng, jenis):
    s = f[f.db == 'sddb']
    vf = s.groupby('rec').apply(lambda g: g.menit_sebelum_vf.max() + 5 + g.mulai_s.min() / 60)
    baris = []
    for k in range(UNDIAN_PALSU):
        pos = []
        for rec, g in s.groupby('rec'):
            T = g.mulai_s.max() / 60 + 5 + g.menit_sebelum_vf.min()   # menit awitan sebenarnya
            if T < 95:
                continue
            tf = rng.uniform(35, T - 60)
            akhir = g.mulai_s / 60 + 5
            w = g[(akhir <= tf) & (akhir > tf - 30)]
            if len(w) >= 3:
                pos.append(w)
        pos = pd.concat(pos)
        for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
            sc, subj = skor_subjek(pos, neg, 'semua', jenis)
            baris.append({'uji': 'palsu', 'model': jenis, 'kontrol': nama, 'undian': k,
                          'n_pos': int(subj.sum()), 'auc': roc_auc_score(subj, sc)})
        print(f'palsu {jenis} undian {k:2d}  lawan nsrdb {baris[-2]["auc"]:.3f}  lawan chfdb {baris[-1]["auc"]:.3f}', flush=True)
    return baris


def uji_tren(f, nsr, chf, rng, jenis):
    cukup = None
    for h in HOR_TETAP:
        r = set(u.ambil_scd(f, h).rec)
        cukup = r if cukup is None else cukup & r
    cukup = sorted(cukup)
    baris = []
    for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
        skor = {}
        for h in HOR_TETAP:
            pos = u.ambil_scd(f, h)
            pos = pos[pos.rec.isin(cukup)]
            skor[h] = skor_subjek(pos, neg, 'semua', jenis)
        subj = skor[0][1]
        idx = subj.index.to_numpy()
        y = subj.to_numpy()
        S = np.column_stack([skor[h][0].reindex(idx).to_numpy() for h in HOR_TETAP])
        jam = np.array(HOR_TETAP) / 60

        def ringkas(ii):
            a = np.array([roc_auc_score(y[ii], S[ii, j]) for j in range(len(HOR_TETAP))])
            return a, np.polyfit(jam, a, 1)[0], a[0] - a[-1]

        a0, m0, d0 = ringkas(np.arange(len(y)))
        pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
        ms, ds = [], []
        for _ in range(u.BOOT):
            ii = np.r_[rng.choice(pos_i, len(pos_i)), rng.choice(neg_i, len(neg_i))]
            _, m, dd = ringkas(ii)
            ms.append(m)
            ds.append(dd)
        r = {'uji': 'tren', 'model': jenis, 'kontrol': nama, 'n_pos': len(pos_i), 'n_neg': len(neg_i),
             'kemiringan_per_jam': m0, 'kem_lo': np.percentile(ms, 2.5), 'kem_hi': np.percentile(ms, 97.5),
             'selisih_0_360': d0, 'sel_lo': np.percentile(ds, 2.5), 'sel_hi': np.percentile(ds, 97.5)}
        r.update({f'auc_{h}': a for h, a in zip(HOR_TETAP, a0)})
        baris.append(r)
        print(f'tren {jenis} lawan {nama}: n={len(pos_i)}/{len(neg_i)}  AUC ' + ' '.join(f'{a:.2f}' for a in a0)
              + f'  kemiringan {m0:+.3f}/jam [{r["kem_lo"]:+.3f}, {r["kem_hi"]:+.3f}]'
              + f'  AUC(0)-AUC(6j) {d0:+.3f} [{r["sel_lo"]:+.3f}, {r["sel_hi"]:+.3f}]', flush=True)
    return baris


def main():
    f = u.siapkan(pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str}))
    rng = np.random.default_rng(2026)
    nsr = u.ambil_kontrol(f, 'nsrdb', rng)
    chf = u.ambil_kontrol(f, 'chfdb', rng)
    baris = []
    for jenis in ('logistik', 'rf'):
        baris += uji_tren(f, nsr, chf, rng, jenis)
    for jenis in ('logistik', 'rf'):
        baris += uji_palsu(f, nsr, chf, rng, jenis)
    h = pd.DataFrame(baris)
    h.to_csv(os.path.join(OUT, 'hasil_uji_lanjut.csv'), index=False)
    p = h[h.uji == 'palsu'].groupby(['model', 'kontrol']).auc.agg(['mean', 'min', 'max'])
    print('\nAwitan palsu, 20 undian:\n', p.round(3))


if __name__ == '__main__':
    main()

"""Artikel EKG 1 — uji dalam-pasien: apakah jendela dekat VF berbeda dari jendela jauh PASIEN YANG SAMA?

Menjawab keberatan efek langit-langit: dengan AUC antarbasis-data ~0,95, kurva horizon yang
datar tidak menyingkirkan perubahan pra-VF yang sesungguhnya. Di sini kelas berasal dari pasien
yang sama, jadi identitas pasien dan basis data tidak bisa membedakan kelas.

  Positif  jendela sddb yang berakhir 0-30 menit sebelum VF (minimal 3 per pasien)
  Negatif  jendela sddb pasien yang sama yang berakhir >= JAUH menit sebelum VF (maks 6 acak)

Pembagian StratifiedGroupKFold 5 lipatan, grup = pasien, 10 ulangan. AUC atas jendela, selang
95 % bootstrap pasien (skor out-of-fold tetap, tanpa latih ulang). Diuji untuk JAUH 120, 180, 360.

Keluaran: keluaran/hasil_dalam_pasien.csv dan ringkasan di layar.
Pakai:    python 07_dalam_pasien.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
CIRI = ['mean_nn', 'sdnn', 'rmssd', 'pnn50', 'cv_nn', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2', 'frac_ektopik']
LOG = ['sdnn', 'rmssd', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2']
JAUH = [120, 180, 360]
ULANG, BOOT, PER = 10, 1000, 6


def model(j):
    if j == 'logistik':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)


def oof(d, j, ciri):
    X, y, g = d[ciri].to_numpy(), d.y.to_numpy(), d.rec.to_numpy()
    ps = []
    for b in range(ULANG):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
            p[te] = model(j).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def boot(d, p, rng):
    y, g = d.y.to_numpy(), d.rec.to_numpy()
    idx = {s: np.where(g == s)[0] for s in np.unique(g)}
    bs = []
    for _ in range(BOOT):
        i = np.concatenate([idx[s] for s in rng.choice(list(idx), len(idx))])
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    f = pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str})
    f = f[f.db == 'sddb'].copy()
    for c in LOG:
        f[c] = np.log(f[c].clip(lower=1e-8))
    f['lf_hf'] = f['lf_hf'].fillna(f['lf_hf'].median())
    rng = np.random.default_rng(2026)
    pos = f[f.menit_sebelum_vf < 30]
    n = pos.groupby('rec').size()
    pos = pos[pos.rec.isin(n[n >= 3].index)]
    baris = []
    for jauh in JAUH:
        neg = f[(f.menit_sebelum_vf >= jauh) & f.rec.isin(pos.rec)]
        neg = neg.sample(frac=1, random_state=int(rng.integers(1 << 30))).groupby('rec').head(PER)
        d = pd.concat([pos[pos.rec.isin(neg.rec)].assign(y=1), neg.assign(y=0)]).reset_index(drop=True)
        for j in ('logistik', 'rf'):
            for nama, ciri in (('semua', CIRI), ('tanpa_ektopik', CIRI[:-1])):
                a, lo, hi = boot(d, oof(d, j, ciri), rng)
                baris.append(dict(jauh=jauh, model=j, ciri=nama, auc=a, lo=lo, hi=hi,
                                  n_pasien=d.rec.nunique(), n_pos=int(d.y.sum()), n_neg=int((1 - d.y).sum())))
                print(f'jauh>={jauh:3d} {j:8s} {nama:13s} AUC {a:.3f} [{lo:.2f},{hi:.2f}] '
                      f'pasien={d.rec.nunique()} jendela {int(d.y.sum())}/{int((1 - d.y).sum())}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_dalam_pasien.csv'), index=False)


if __name__ == '__main__':
    main()

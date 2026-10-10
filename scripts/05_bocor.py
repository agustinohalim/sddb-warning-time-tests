"""Artikel EKG 1 — replikasi rancangan bocor dari pustaka, pada data dan ciri yang sama.

Rancangan Lai dkk. 2019 (IEEE Access) dan Reddy & Kumar 2022 (IJECE): 30 segmen satu menit per
rekaman — SCD 30 menit tepat sebelum VF, normal 30 menit dari tengah rekaman nsrdb — lalu
validasi silang 5 lipatan ATAS SEGMEN. Di sini rancangan itu dijalankan dua kali:

  bocor  KFold acak atas segmen, persis seperti naskahnya
  benar  StratifiedGroupKFold, grup = rekaman

Pengklasifikasi yang dilaporkan terbaik oleh kedua naskah: random forest (Lai, 99,49 %) dan KNN
(Reddy, 100 %); ditambah SVM RBF. Ciri HRV sama dengan 02_fitur.py, dihitung per segmen satu
menit (ambang minimal interval NN diturunkan ke 40 karena segmennya lima kali lebih pendek).

Keluaran: keluaran/hasil_bocor.csv dan ringkasan di layar.
Pakai:    python 05_bocor.py
"""
import os
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

fit = import_module('02_fitur')
u = import_module('03_uji')
OUT = u.OUT
fit.MIN_NN = 40
ULANG = 10


def segmen():
    den = pd.read_csv(os.path.join(OUT, 'denyut.csv'), dtype={'rec': str})
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    baris = []
    for _, r in rek[rek.db.isin(['sddb', 'nsrdb'])].iterrows():
        if r.db == 'sddb' and pd.isna(r.vf_s):
            continue
        g = den[(den.db == r.db) & (den.rec == r.rec)]
        t, sym = g.t.to_numpy(), g.sym.to_numpy().astype(str)
        mulai = r.vf_s - 1800 if r.db == 'sddb' else r.durasi_s / 2 - 900
        for k in range(30):
            a = mulai + 60 * k
            m = (t >= a) & (t < a + 60)
            c = fit.ciri(t[m], sym[m])
            if c is not None:
                c.update({'db': r.db, 'rec': r.rec, 'y': int(r.db == 'sddb')})
                baris.append(c)
    return u.siapkan(pd.DataFrame(baris))


def model(nama):
    if nama == 'rf':
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    if nama == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    return make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))


def main():
    d = segmen()
    X, y, g = d[u.CIRI_SEMUA].to_numpy(), d.y.to_numpy(), (d.db + ':' + d.rec).to_numpy()
    print(f'{len(d)} segmen satu menit dari {len(set(g))} rekaman (SCD {d[d.y == 1].rec.nunique()}, normal {d[d.y == 0].rec.nunique()})')
    baris = []
    for nama in ('rf', 'knn', 'svm'):
        for skema in ('bocor', 'benar'):
            acc, auc, auc_subj = [], [], []
            for b in range(ULANG):
                p = np.zeros(len(y))
                cv = (KFold(5, shuffle=True, random_state=b).split(X) if skema == 'bocor'
                      else StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g))
                for tr, te in cv:
                    p[te] = model(nama).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                acc.append(accuracy_score(y, p > 0.5))
                auc.append(roc_auc_score(y, p))
                ps = pd.Series(p).groupby(g).mean()
                ys = pd.Series(y).groupby(g).first()
                auc_subj.append(roc_auc_score(ys, ps))
            r = {'model': nama, 'skema': skema, 'akurasi_segmen': np.mean(acc), 'akurasi_min': np.min(acc),
                 'akurasi_max': np.max(acc), 'auc_segmen': np.mean(auc), 'auc_subjek': np.mean(auc_subj)}
            baris.append(r)
            print(f'{nama:4s} {skema:6s} akurasi segmen {r["akurasi_segmen"] * 100:6.2f} %'
                  f' [{r["akurasi_min"] * 100:.1f}-{r["akurasi_max"] * 100:.1f}]  AUC segmen {r["auc_segmen"]:.3f}'
                  f'  AUC subjek {r["auc_subjek"]:.3f}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_bocor.csv'), index=False)


if __name__ == '__main__':
    main()

"""Artikel EKG 1 — apakah "prediksi SCD" sddb lawan nsrdb meramal kematian atau membedakan basis data?

Seluruh pembagian tingkat subjek: StratifiedGroupKFold 5 lipatan, grup = rekaman, diulang 20
kali dengan benih berbeda. Skor subjek = rerata peluang atas jendelanya (maks. 6 jendela 5
menit). AUC dihitung atas subjek, selang 95 % dari bootstrap subjek (1.000 undian).

Uji:
  H  horizon: SCD dengan jendela yang berakhir h menit sebelum VF, lawan nsrdb.
     Bila AUC nyaris sama pada h = 0 dan h = 6 jam, pengklasifikasi tidak melihat kematian
     yang mendekat.
  K  kontrol sakit tetapi hidup: SCD(h) lawan chfdb. chfdb direkam 250 Hz seperti sddb, jadi
     frekuensi cuplik tidak lagi membedakan kelas.
  C  model yang dilatih SCD lawan nsrdb dipakai pada pasien chfdb: berapa bagian yang
     dinyatakan "akan meninggal"?

Set ciri: semua; semua tanpa frac_ektopik; frac_ektopik saja; mean_nn saja (laju jantung).

Keluaran: keluaran/hasil_uji.csv dan ringkasan di layar.
Pakai:    python 03_uji.py
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
HORIZON = [0, 30, 60, 120, 180, 360, 720]      # menit sebelum VF, awal pita 30 menit
PER_SUBJEK = 6
ULANG = 20
BOOT = 1000

CIRI_SEMUA = ['mean_nn', 'sdnn', 'rmssd', 'pnn50', 'cv_nn', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2', 'frac_ektopik']
SET = {
    'semua': CIRI_SEMUA,
    'tanpa_ektopik': [c for c in CIRI_SEMUA if c != 'frac_ektopik'],
    'ektopik_saja': ['frac_ektopik'],
    'laju_saja': ['mean_nn'],
}
POSITIF = ['sdnn', 'rmssd', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2']   # miring, dilog


def siapkan(f):
    f = f.copy()
    for c in POSITIF:
        f[c] = np.log(f[c].clip(lower=1e-8))
    f['lf_hf'] = f['lf_hf'].fillna(f['lf_hf'].median())
    return f


def ambil_kontrol(f, db, rng):
    g = f[f.db == db]
    return g.groupby('rec').sample(n=PER_SUBJEK, random_state=int(rng.integers(1 << 30)))


def ambil_scd(f, h):
    g = f[(f.db == 'sddb') & (f.menit_sebelum_vf >= h) & (f.menit_sebelum_vf < h + 30)]
    n = g.groupby('rec').size()
    return g[g.rec.isin(n[n >= 3].index)]


def model(jenis):
    if jenis == 'logistik':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)


def skor_cv(d, ciri, jenis, benih, luar=None):
    """Skor subjek di luar lipatan; bila 'luar' diberi, subjek itu diskor oleh model tiap lipatan."""
    subj = d.groupby('rec').y.first()
    X, y, g = d[ciri].to_numpy(), d.y.to_numpy(), d.rec.to_numpy()
    p = np.zeros(len(d))
    pl = [] if luar is not None else None
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=benih).split(X, y, g):
        m = model(jenis).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        if luar is not None:
            pl.append(m.predict_proba(luar[ciri].to_numpy())[:, 1])
    s = pd.Series(p).groupby(d.rec.to_numpy()).mean()
    sl = None
    if luar is not None:
        sl = pd.Series(np.mean(pl, axis=0)).groupby(luar.rec.to_numpy()).mean()
    return s.reindex(subj.index), subj, sl


def auc_ci(y, s, rng):
    y, s = np.asarray(y), np.asarray(s)
    a = roc_auc_score(y, s)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def uji(f, pos, neg, nama, ciri_set, jenis, rng, luar=None):
    d = pd.concat([pos.assign(y=1), neg.assign(y=0)])
    d = d.assign(rec=d.db + ':' + d.rec)
    if luar is not None:
        luar = luar.assign(rec=luar.db + ':' + luar.rec)
    skor, skor_luar = [], []
    for b in range(ULANG):
        s, subj, sl = skor_cv(d, SET[ciri_set], jenis, b, luar)
        skor.append(s)
        if sl is not None:
            skor_luar.append(sl)
    s = pd.concat(skor, axis=1).mean(axis=1)
    a, lo, hi = auc_ci(subj.to_numpy(), s.to_numpy(), rng)
    hasil = {'uji': nama, 'ciri': ciri_set, 'model': jenis, 'n_pos': int(subj.sum()),
             'n_neg': int((1 - subj).sum()), 'auc': a, 'lo': lo, 'hi': hi}
    if skor_luar:
        sl = pd.concat(skor_luar, axis=1).mean(axis=1)
        hasil['frac_chf_dinyatakan_scd'] = float((sl > 0.5).mean())
        hasil['auc_scd_lawan_chf'] = roc_auc_score(
            np.r_[np.ones(int(subj.sum())), np.zeros(len(sl))], np.r_[s[subj == 1].to_numpy(), sl.to_numpy()])
    return hasil


def main():
    f = siapkan(pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str}))
    rng = np.random.default_rng(2026)
    nsr = ambil_kontrol(f, 'nsrdb', rng)
    chf = ambil_kontrol(f, 'chfdb', rng)
    baris = []
    for jenis in ('logistik', 'rf'):
        for cs in SET:
            for h in HORIZON:
                scd = ambil_scd(f, h)
                luar = chf if h == 0 else None
                r = uji(f, scd, nsr, f'H_nsr_{h}', cs, jenis, rng, luar)
                r['horizon_menit'] = h
                baris.append(r)
                r2 = uji(f, scd, chf, f'K_chf_{h}', cs, jenis, rng)
                r2['horizon_menit'] = h
                baris.append(r2)
                print(f'{jenis:8s} {cs:14s} h={h:4d}  SCD-NSR AUC {r["auc"]:.3f} [{r["lo"]:.2f},{r["hi"]:.2f}] n={r["n_pos"]}/{r["n_neg"]}'
                      f'   SCD-CHF AUC {r2["auc"]:.3f} [{r2["lo"]:.2f},{r2["hi"]:.2f}]'
                      + (f'   CHF->"SCD" {r["frac_chf_dinyatakan_scd"]:.2f}' if h == 0 else ''), flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_uji.csv'), index=False)


if __name__ == '__main__':
    main()

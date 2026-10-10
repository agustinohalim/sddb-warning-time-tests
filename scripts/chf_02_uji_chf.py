"""Artikel EKG 3 — apakah deteksi gagal jantung dari HRV pada data PhysioNet mengenali penyakit atau basis data?

  T1 rancangan baku     CHF (chfdb + chf2db) lawan sehat (nsrdb + nsr2db), pembagian subjek
  T2 probe identitas    di dalam kelas yang sama: chfdb lawan chf2db; nsrdb lawan nsr2db;
                        fantasia muda lawan nsrdb (sama-sama sehat dan muda). Gagasan probe dari
                        Zare (2026) untuk EEG; di sini diterapkan pada HRV.
  T3 antarbasis data    latih chfdb lawan nsrdb, uji chf2db lawan nsr2db; dan sebaliknya
  T4 usia               model T1 (dilatih pada seluruh data T1) diterapkan pada Fantasia: berapa
                        bagian orang sehat tua dan muda dinyatakan CHF

Skor subjek = rerata peluang atas maksimal 6 jendela acak; AUC atas subjek; selang 95 % bootstrap
subjek. Model logistik dan random forest; StratifiedGroupKFold 5 lipatan x 10 ulangan.

Keluaran: keluaran/hasil_chf.csv dan ringkasan di layar.
Pakai:    python 02_uji_chf.py
"""
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
sys.path.insert(0, HERE)
u = import_module('af_02_uji_af')
PER = 6


def pasang(a, b):
    return pd.concat([a.assign(y=1), b.assign(y=0)])


def uji_cv(d, j, rng):
    return u.boot_auc(d, u.oof(d, j), 'subjek', rng)


def skor_subjek(m, d):
    p = m.predict_proba(d[u.CIRI].to_numpy())[:, 1]
    return pd.Series(p).groupby(d.subj.to_numpy()).mean()


def main():
    f = u.siapkan(pd.read_csv(os.path.join(OUT, 'fitur_chf.csv'), dtype={'rec': str}))
    rng = np.random.default_rng(2026)
    s = u.ambil(f, PER, rng)
    db = {k: s[s.db == k] for k in s.db.unique()}
    fan = s[s.db == 'fantasia']
    baris = []

    def catat(nama, j, r, **kw):
        baris.append(dict(uji=nama, model=j, auc=r[0], lo=r[1], hi=r[2], n_pos=r[3], n_neg=r[4], **kw))
        print(f'{nama:32s} {j:8s} AUC {r[0]:.3f} [{r[1]:.2f},{r[2]:.2f}] n={r[3]}/{r[4]}', flush=True)

    for j in ('logistik', 'rf'):
        catat('T1_baku_CHF_lawan_sehat', j, uji_cv(pasang(pd.concat([db['chfdb'], db['chf2db']]),
                                                          pd.concat([db['nsrdb'], db['nsr2db']])), j, rng))
        catat('T2_probe_chfdb_lawan_chf2db', j, uji_cv(pasang(db['chfdb'], db['chf2db']), j, rng))
        catat('T2_probe_nsrdb_lawan_nsr2db', j, uji_cv(pasang(db['nsrdb'], db['nsr2db']), j, rng))
        catat('T2_probe_fantasiamuda_lawan_nsrdb', j, uji_cv(pasang(fan[fan.kelas == 'muda'], db['nsrdb']), j, rng))
        for latih, uji in ((('chfdb', 'nsrdb'), ('chf2db', 'nsr2db')), (('chf2db', 'nsr2db'), ('chfdb', 'nsrdb'))):
            a = pasang(db[latih[0]], db[latih[1]])
            b = pasang(db[uji[0]], db[uji[1]])
            m = u.model(j).fit(a[u.CIRI].to_numpy(), a.y.to_numpy())
            sc = skor_subjek(m, b)
            yy = b.groupby('subj').y.first().reindex(sc.index).to_numpy()
            bs = [roc_auc_score(yy[i], sc.to_numpy()[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(u.BOOT)) if 0 < yy[i].sum() < len(i)]
            catat(f'T3_latih_{latih[0]}+{latih[1]}_uji_{uji[0]}+{uji[1]}', j,
                  (roc_auc_score(yy, sc), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(yy.sum()), int((1 - yy).sum())))
        t1 = pasang(pd.concat([db['chfdb'], db['chf2db']]), pd.concat([db['nsrdb'], db['nsr2db']]))
        m = u.model(j).fit(t1[u.CIRI].to_numpy(), t1.y.to_numpy())
        sf = skor_subjek(m, fan)
        kel = fan.groupby('subj').kelas.first().reindex(sf.index)
        tua, muda = sf[kel == 'tua'], sf[kel == 'muda']
        r = dict(uji='T4_fantasia', model=j, frac_tua_dinyatakan_CHF=float((tua > 0.5).mean()),
                 frac_muda_dinyatakan_CHF=float((muda > 0.5).mean()),
                 auc_tua_lawan_muda=roc_auc_score(np.r_[np.ones(len(tua)), np.zeros(len(muda))], np.r_[tua, muda]))
        baris.append(r)
        print(f'T4 {j:8s} Fantasia tua dinyatakan CHF {r["frac_tua_dinyatakan_CHF"]:.2f}, muda {r["frac_muda_dinyatakan_CHF"]:.2f};'
              f' AUC skor-CHF tua lawan muda {r["auc_tua_lawan_muda"]:.3f}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_chf.csv'), index=False)
    print('\nusia per basis data (dari header):', f.groupby('db').usia.agg(lambda x: pd.to_numeric(x, errors="coerce").describe()[['mean', 'min', 'max']].round(1).to_dict()).to_dict())


if __name__ == '__main__':
    main()

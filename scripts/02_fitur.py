"""Artikel EKG 1 — ciri HRV per jendela 5 menit, untuk seluruh rekaman.

Jendela 5 menit tidak bertumpang, sesuai pedoman HRV Task Force 1996 dan sesuai mayoritas
pustaka yang diaudit. Untuk sddb, hanya jendela yang selesai sebelum awitan VF yang disimpan,
dan setiap jendela diberi 'menit_sebelum_vf' = jarak ujung jendela ke awitan.

Ciri:
  waktu      mean_nn, sdnn, rmssd, pnn50, cv_nn
  frekuensi  lf, hf, lf_hf (Lomb-Scargle atas deret NN yang tidak berjarak sama)
  Poincare   sd1, sd2
  ektopik    frac_ektopik — proporsi denyut bukan normal. Sengaja disimpan terpisah: pasien
             Holter yang sakit punya banyak denyut ektopik, orang sehat hampir tidak, jadi ciri
             ini menjadi kandidat jalan pintas yang paling jelas.

Interval NN = selisih dua denyut 'N' berurutan dengan 0,3 s < RR < 2 s. Jendela dengan kurang
dari 150 interval NN dibuang (derau, sinyal hilang).

Keluaran: keluaran/fitur.csv
Pakai:    python 02_fitur.py
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
JENDELA = 300.0
DENYUT = set('NLRBAaJSVrFejnE/fQ')   # simbol denyut WFDB; sisanya penanda derau/irama
MIN_NN = 150
FREK = np.linspace(0.01, 0.5, 200)


def ciri(t, sym):
    beat = np.isin(sym, list(DENYUT))
    t, sym = t[beat], sym[beat]
    if len(t) < MIN_NN:
        return None
    rr = np.diff(t)
    ok = (sym[1:] == 'N') & (sym[:-1] == 'N') & (rr > 0.3) & (rr < 2.0)
    nn, tn = rr[ok], t[1:][ok]
    if len(nn) < MIN_NN:
        return None
    d = np.diff(nn)
    x = nn - nn.mean()
    p = lombscargle(tn - tn[0], x, 2 * np.pi * FREK, normalize=False)
    df = FREK[1] - FREK[0]
    lf = p[(FREK >= 0.04) & (FREK < 0.15)].sum() * df
    hf = p[(FREK >= 0.15) & (FREK < 0.4)].sum() * df
    sd1 = np.sqrt(np.var(d) / 2)
    sd2 = np.sqrt(max(2 * np.var(nn) - np.var(d) / 2, 0))
    return {
        'n_nn': len(nn), 'mean_nn': nn.mean(), 'sdnn': nn.std(), 'rmssd': np.sqrt(np.mean(d ** 2)),
        'pnn50': np.mean(np.abs(d) > 0.05), 'cv_nn': nn.std() / nn.mean(),
        'lf': lf, 'hf': hf, 'lf_hf': lf / hf if hf > 0 else np.nan,
        'sd1': sd1, 'sd2': sd2, 'frac_ektopik': np.mean(sym != 'N'),
    }


def main():
    den = pd.read_csv(os.path.join(OUT, 'denyut.csv'), dtype={'rec': str})
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    baris = []
    for _, r in rek.iterrows():
        g = den[(den.db == r.db) & (den.rec == r.rec)]
        t, sym = g.t.to_numpy(), g.sym.to_numpy().astype(str)
        akhir = r.vf_s if r.db == 'sddb' else r.durasi_s
        if r.db == 'sddb' and pd.isna(r.vf_s):
            continue   # tiga rekaman tanpa VF (40, 42, 49) dibuang, seperti pustakanya
        for a in np.arange(0, akhir - JENDELA + 1, JENDELA):
            m = (t >= a) & (t < a + JENDELA)
            c = ciri(t[m], sym[m])
            if c is None:
                continue
            c.update({'db': r.db, 'rec': r.rec, 'mulai_s': a,
                      'menit_sebelum_vf': (akhir - (a + JENDELA)) / 60 if r.db == 'sddb' else np.nan})
            baris.append(c)
        print(f'{r.db:6s} {r.rec:6s} {sum(1 for b in baris if b["rec"] == r.rec and b["db"] == r.db):4d} jendela')
    f = pd.DataFrame(baris)
    f.to_csv(os.path.join(OUT, 'fitur.csv'), index=False)
    print(f'\n{len(f)} jendela; per basis data:', f.groupby('db').size().to_dict())


if __name__ == '__main__':
    main()

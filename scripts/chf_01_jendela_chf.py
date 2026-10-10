"""Artikel EKG 3 — jendela HRV 5 menit dari lima basis data PhysioNet untuk deteksi gagal jantung.

Kelas gagal jantung (CHF):
  chfdb     15 subjek, NYHA III-IV, 250 Hz, anotasi .ecg (sudah diunduh untuk artikel 1)
  chf2db    29 subjek, NYHA I-III, 128 Hz, anotasi .ecg
Kelas sehat:
  nsrdb     18 subjek, 20-50 th, 128 Hz, anotasi .atr (Thesis/Data/nsrdb)
  nsr2db    54 subjek, 28-76 th, 128 Hz, anotasi .ecg (sudah diunduh untuk artikel 2)
Kontrol usia:
  fantasia  20 muda (21-34 th) dan 20 tua (68-85 th), sehat, 250 Hz, sekitar 2 jam, anotasi .ecg

Semua terbuka, ODC-By 1.0. Ciri sama dengan artikel 2: 10 ciri HRV dari semua jarak RR antar denyut
berurutan (0,3 s < RR < 2 s), tanpa proporsi ektopik — tidak semua basis data memuat jenis denyut
yang sebanding.

Keluaran: keluaran/fitur_chf.csv
Pakai:    python 01_jendela_chf.py
"""
import os
from importlib import import_module

import numpy as np
import pandas as pd
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
EKG = os.path.join(HERE, '..')
OUT = os.path.join(HERE, 'keluaran')
DATA = os.path.join(HERE, '..', 'data')
JENDELA = 300.0
DENYUT = list('NLRBAaJSVrFejnE/fQ')

import sys
sys.path.insert(0, HERE)
ciri = import_module('af_01_jendela').ciri

SUMBER = {
    'chfdb': (os.path.join(os.path.join(HERE, '..', 'data'), 'chfdb'), 'ecg', 'CHF'),
    'chf2db': (os.path.join(DATA, 'chf2db'), 'ecg', 'CHF'),
    'nsrdb': (os.path.join(os.path.join(HERE, '..', 'data'), 'nsrdb'), 'atr', 'sehat'),
    'nsr2db': (os.path.join(os.path.join(HERE, '..', 'data'), 'nsr2db'), 'ecg', 'sehat'),
    'fantasia': (os.path.join(DATA, 'fantasia'), 'ecg', 'usia'),
}


def unduh(db, folder, ext):
    os.makedirs(folder, exist_ok=True)
    rek = wfdb.get_record_list(db)
    perlu = [f'{r}.{e}' for r in rek for e in ('hea', ext) if not os.path.exists(os.path.join(folder, f'{r}.{e}'))]
    for f in perlu:
        try:
            wfdb.dl_files(db, folder, [f])
        except Exception as e:
            print('gagal unduh', db, f, e)


def main():
    os.makedirs(OUT, exist_ok=True)
    for db in ('chf2db', 'fantasia'):
        unduh(db, SUMBER[db][0], SUMBER[db][1])
    baris = []
    for db, (folder, ext, kelas) in SUMBER.items():
        for r in sorted({p[:-4] for p in os.listdir(folder) if p.endswith('.hea')}):
            path = os.path.join(folder, r)
            if not os.path.exists(f'{path}.{ext}'):
                continue
            h = wfdb.rdheader(path)
            a = wfdb.rdann(path, ext)
            sym = np.array(a.symbol)
            t = a.sample[np.isin(sym, DENYUT)] / a.fs if a.fs else a.sample[np.isin(sym, DENYUT)] / h.fs
            dur = h.sig_len / h.fs if h.sig_len else t.max()
            usia = ''
            for c in h.comments:
                for tok in c.replace(',', ' ').split():
                    if tok.isdigit() and 10 < int(tok) < 100:
                        usia = int(tok)
                        break
                if usia != '':
                    break
            n = 0
            for s in np.arange(0, dur - JENDELA + 1, JENDELA):
                c = ciri(t[(t >= s) & (t < s + JENDELA)])
                if c is None:
                    continue
                c.update({'db': db, 'rec': r, 'kelas': kelas, 'fs': h.fs, 'usia': usia, 'mulai_s': s})
                if db == 'fantasia':
                    c['kelas'] = 'tua' if 'o' in r[2:3] else 'muda'
                baris.append(c)
                n += 1
            print(f'{db:9s} {r:7s} fs={h.fs:5.0f} usia={usia!s:4s} {n:4d} jendela', flush=True)
    f = pd.DataFrame(baris)
    f.to_csv(os.path.join(OUT, 'fitur_chf.csv'), index=False)
    print(f'\n{len(f)} jendela;', f.groupby('db').rec.nunique().to_dict(), 'subjek')


if __name__ == '__main__':
    main()

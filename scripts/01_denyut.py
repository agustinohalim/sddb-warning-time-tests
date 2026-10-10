"""Artikel EKG 1 — kumpulkan denyut dari tiga basis data PhysioNet.

Hanya anotasi denyut yang dibaca (waktu dan jenis denyut), bukan sinyal mentah: seluruh uji
artikel ini memakai ciri HRV, sama dengan mayoritas pustaka yang diaudit.

  sddb   23 rekaman Holter pasien yang mengalami henti jantung. Anotasi .ari (belum diaudit)
         dipakai untuk semua rekaman supaya seragam; .atr teraudit hanya ada untuk 9.
  nsrdb  18 rekaman subjek sehat, anotasi .atr.
  chfdb  15 rekaman gagal jantung berat yang TIDAK meninggal selama rekaman — kontrol "sakit
         tetapi hidup". Anotasi .ecg diunduh dari PhysioNet (terbuka, ODC-By), tanpa sinyal.

Waktu awitan VF sddb diambil dari halaman PhysioNet (salinannya di Thesis/Data/VF Data.csv).

Keluaran: keluaran/denyut.csv (db, rec, t detik, sym) dan keluaran/rekaman.csv.
Pakai:    python 01_denyut.py
"""
import csv
import os

import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(HERE, '..', 'data')
CHF = os.path.join(os.path.join(HERE, '..', 'data'), 'chfdb')
OUT = os.path.join(HERE, 'keluaran')


def hms(s):
    h, m, d = (int(x) for x in s.split('.'))
    return h * 3600 + m * 60 + d


def vf_onset():
    hasil = {}
    with open(os.path.join(THESIS, 'VF Data.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter=';'):
            rec = r['Dat Files'].split('.')[0]
            v = r['VF Onset Time (elapsed)']
            hasil[rec] = hms(v) if v[0].isdigit() else None
    return hasil


def unduh_chf():
    os.makedirs(CHF, exist_ok=True)
    nama = [f'chf{i:02d}' for i in range(1, 16)]
    perlu = [f'{n}.{e}' for n in nama for e in ('hea', 'ecg')
             if not os.path.exists(os.path.join(CHF, f'{n}.{e}'))]
    if perlu:
        wfdb.dl_files('chfdb', CHF, perlu)
    return nama


def main():
    os.makedirs(OUT, exist_ok=True)
    vf = vf_onset()
    sumber = ([('sddb', r, os.path.join(THESIS, 'sddb', r), 'ari') for r in sorted(vf)]
              + [('nsrdb', os.path.basename(p)[:-4], os.path.join(THESIS, 'nsrdb', os.path.basename(p)[:-4]), 'atr')
                 for p in sorted(os.listdir(os.path.join(THESIS, 'nsrdb'))) if p.endswith('.hea')]
              + [('chfdb', n, os.path.join(CHF, n), 'ecg') for n in unduh_chf()])
    sumber = [s for s in sumber if not s[1].endswith('.hea')]
    with open(os.path.join(OUT, 'denyut.csv'), 'w', newline='') as f1, \
            open(os.path.join(OUT, 'rekaman.csv'), 'w', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['db', 'rec', 't', 'sym'])
        w2.writerow(['db', 'rec', 'fs', 'durasi_s', 'vf_s', 'n_denyut'])
        for db, rec, path, ext in sumber:
            h = wfdb.rdheader(path)
            a = wfdb.rdann(path, ext)
            for s, sym in zip(a.sample, a.symbol):
                w1.writerow([db, rec, round(s / h.fs, 4), sym])
            w2.writerow([db, rec, h.fs, round(h.sig_len / h.fs, 1), vf.get(rec, '') if db == 'sddb' else '',
                         len(a.sample)])
            print(f'{db:6s} {rec:6s} fs={h.fs:5.0f} {h.sig_len / h.fs / 3600:5.1f} jam  {len(a.sample):7d} anotasi')


if __name__ == '__main__':
    main()

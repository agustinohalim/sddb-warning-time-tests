"""Download the PhysioNet annotation files used by the analyses into ../data/.

Only headers and beat or rhythm annotations are downloaded, no raw signals: every analysis in the
paper uses beat times and labels. About 60 MB in total.

  sddb     .hea, .ari (automated, all records), .atr (audited, where it exists)
  nsrdb    .hea, .atr
  chfdb    .hea, .ecg
  chf2db   .hea, .ecg          (heart failure positive control)
  nsr2db   .hea, .ecg          (heart failure positive control)
  fantasia .hea, .ecg          (heart failure positive control)

All databases are open under the Open Data Commons Attribution License v1.0.
Usage: python 00_get_data.py
"""
import os

import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
PLAN = {
    'sddb': ['hea', 'ari', 'atr'],
    'nsrdb': ['hea', 'atr'],
    'chfdb': ['hea', 'ecg'],
    'chf2db': ['hea', 'ecg'],
    'nsr2db': ['hea', 'ecg'],
    'fantasia': ['hea', 'ecg'],
}


def main():
    for db, ext in PLAN.items():
        tujuan = os.path.join(DATA, db)
        os.makedirs(tujuan, exist_ok=True)
        recs = wfdb.get_record_list(db)
        n = 0
        for r in recs:
            for e in ext:
                f = f'{r}.{e}'
                if os.path.exists(os.path.join(tujuan, f)):
                    continue
                try:
                    wfdb.dl_files(db, tujuan, [f])
                    n += 1
                except Exception:
                    if not (db == 'sddb' and e == 'atr'):   # audited .atr exists for 12 sddb records only
                        raise
        print(f'{db:8s} {len(recs):3d} records, {n} files downloaded', flush=True)


if __name__ == '__main__':
    main()

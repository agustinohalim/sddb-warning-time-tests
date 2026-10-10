"""Artikel EKG 1 — analisis yang diminta telaah ulang panel 8 Oktober 2026 (Telaah/2026-10-08-EKG1/report.md).

Bagian (argumen; tanpa argumen semua, urut):
  irama     tanda per jendela 5 menit dari anotasi irama (aux_note '(AFIB', '(AFL', '(VT', '(VFL', '(P',
            '(B', '(T', '(SVTA', ...): AF/flutter, irama pacu (denyut '/' atau '(P'), VT (aux '(VT' atau
            >= 3 denyut V berturut-turut). -> keluaran/irama_jendela.csv
  palsu     uji surogat awitan palsu yang benar: pasien yang SAMA untuk awitan sebenarnya dan palsu,
            99 undian (logistik) / 39 (RF); p peringkat = (1 + #palsu >= sebenarnya) / (N + 1)
  refit     selang bootstrap dengan latih ulang untuk kecenderungan kohort tetap (AUC(0) - AUC(360))
  daya      simulasi daya: perubahan pra-VF buatan disuntikkan ke jendela sddb (besar delta SD,
            meluruh e^(-t/60 menit)); seberapa sering uji horizon dan uji dalam-pasien menangkapnya
  halus     pita 10 menit dalam 30 menit terakhir (0-10, 10-20, 20-30) lawan nsrdb/chfdb, dan dalam-pasien
  sinus     ulang uji horizon, kohort tetap, dan dalam-pasien hanya dengan jendela tanpa AF/flutter,
            pacu, dan VT; dalam-pasien juga hanya tanpa VT
  pasangan  dalam-pasien berpasangan: per pasien, rerata skor OOF jendela dekat minus jauh; uji tanda
            dan Wilcoxon atas pasien

Anotasi sddb .ari (sama dengan 01), nsrdb .atr, chfdb .ecg. Pembagian dan model sama dengan 03/04/07.
Keluaran: keluaran/revisi_panel.log (lewat >), keluaran/irama_jendela.csv
Pakai:    python -u 09_revisi_panel.py > keluaran/revisi_panel.log
"""
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
v = import_module('04_uji_lanjut')
w = import_module('07_dalam_pasien')
HERE, OUT = u.HERE, u.OUT
THESIS = os.path.join(HERE, '..', 'data')
CIRI = u.CIRI_SEMUA
rng = np.random.default_rng(20261009)


# ---------------------------------------------------------------- irama
def irama():
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    f = pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str})
    baris = []
    for _, r in rek.iterrows():
        path = (os.path.join(THESIS, r.db, r.rec) if r.db != 'chfdb' else os.path.join(os.path.join(HERE, '..', 'data'), 'chfdb', r.rec))
        ext = {'sddb': 'ari', 'nsrdb': 'atr', 'chfdb': 'ecg'}[r.db]
        a = wfdb.rdann(path, ext)
        t = np.asarray(a.sample) / r.fs
        sym = np.asarray(a.symbol)
        aux = [x.strip('\x00') for x in a.aux_note]
        # irama berlaku dari tanda '(X' sampai tanda irama berikutnya
        mulai, jenis = [0.0], ['N']
        for ti, x in zip(t, aux):
            if x.startswith('(') and not x.startswith('(ST') and not x.startswith('(T'):
                mulai.append(ti)
                jenis.append(x[1:])
        mulai, jenis = np.array(mulai), np.array(jenis)
        g = f[(f.db == r.db) & (f.rec == r.rec)]
        for m in g.mulai_s.to_numpy():
            e = m + 300
            i0 = np.searchsorted(mulai, m, side='right') - 1
            i1 = np.searchsorted(mulai, e, side='left')
            rit = set(jenis[max(i0, 0):max(i1, i0 + 1)])
            k = (t >= m) & (t < e)
            s = sym[k]
            run, vt_run = 0, False
            for c in s:
                run = run + 1 if c == 'V' else 0
                vt_run |= run >= 3
            baris.append(dict(db=r.db, rec=r.rec, mulai_s=m,
                              af=any(x.startswith(('AFIB', 'AFL', 'AF')) for x in rit),
                              pacu=bool((s == '/').mean() > 0.01) or any(x.startswith('P') for x in rit),
                              vt=vt_run or any(x.startswith(('VT', 'VFL', 'VF')) for x in rit)))
    d = pd.DataFrame(baris)
    d.to_csv(os.path.join(OUT, 'irama_jendela.csv'), index=False)
    print('== Tanda irama per jendela (proporsi jendela; rekaman dengan tanda apa pun)')
    for db, g in d.groupby('db'):
        rek_t = g.groupby('rec')[['af', 'pacu', 'vt']].any().sum().to_dict()
        print(f'  {db:6s} jendela {len(g):5d}  AF {g.af.mean():.3f}  pacu {g.pacu.mean():.3f}  VT {g.vt.mean():.3f}'
              f'  | rekaman: {rek_t} dari {g.rec.nunique()}', flush=True)


def fitur(saring=None):
    f = u.siapkan(pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str}))
    if saring:
        ir = pd.read_csv(os.path.join(OUT, 'irama_jendela.csv'), dtype={'rec': str})
        f = f.merge(ir, on=['db', 'rec', 'mulai_s'], how='left')
        buang = np.zeros(len(f), bool)
        for k in saring:
            buang |= f[k].fillna(False).to_numpy(bool)
        f = f[~buang].drop(columns=['af', 'pacu', 'vt'])
    return f


def kontrol(f, r):
    return u.ambil_kontrol(f, 'nsrdb', r), u.ambil_kontrol(f, 'chfdb', r)


# ---------------------------------------------------------------- palsu (surogat)
def onset_palsu(s, r, tetap):
    pos_true, pos_f = [], []
    for rec, g in s.groupby('rec'):
        if rec not in tetap:
            continue
        T = g.mulai_s.max() / 60 + 5 + g.menit_sebelum_vf.min()
        tf = r.uniform(35, T - 60)
        akhir = g.mulai_s / 60 + 5
        pos_f.append(g[(akhir <= tf) & (akhir > tf - 30)])
    return pd.concat(pos_f)


def palsu():
    f = fitur()
    r = np.random.default_rng(1)
    nsr, chf = kontrol(f, r)
    s = f[f.db == 'sddb']
    benar = u.ambil_scd(f, 0)
    # pasien yang memenuhi syarat awitan palsu (T >= 95 menit) DAN punya >= 3 jendela tiap undian
    calon = []
    for rec, g in s.groupby('rec'):
        T = g.mulai_s.max() / 60 + 5 + g.menit_sebelum_vf.min()
        if T >= 95 and rec in set(benar.rec):
            calon.append(rec)
    print(f'\n== Uji surogat awitan palsu: {len(calon)} pasien sama untuk awitan sebenarnya dan palsu', flush=True)
    for jenis, N in (('logistik', 99), ('rf', 39)):
        for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
            sc, subj = v.skor_subjek(benar[benar.rec.isin(calon)], neg, 'semua', jenis)
            a_true = roc_auc_score(subj, sc)
            a_f, coba = [], 0
            while len(a_f) < N and coba < 50 * N:
                coba += 1
                pf = onset_palsu(s, r, set(calon))
                n = pf.groupby('rec').size()
                if (n >= 3).sum() < len(calon):          # undian ulang bila ada pasien kurang jendela
                    continue
                sc, subj = v.skor_subjek(pf, neg, 'semua', jenis)
                a_f.append(roc_auc_score(subj, sc))
            a_f = np.array(a_f)
            p = (1 + (a_f >= a_true).sum()) / (N + 1)
            print(f'  {jenis:8s} lawan {nama}: sebenarnya {a_true:.3f}; palsu rerata {a_f.mean():.3f} '
                  f'[{a_f.min():.3f}, {a_f.max():.3f}]; selisih {a_true - a_f.mean():+.3f}; p peringkat {p:.3f} (N={N})',
                  flush=True)


# ---------------------------------------------------------------- refit bootstrap
def kohort_tetap(f):
    c = None
    for h in v.HOR_TETAP:
        r_ = set(u.ambil_scd(f, h).rec)
        c = r_ if c is None else c & r_
    return sorted(c)


def auc_horizon_refit(f, neg, tetap_ids, jenis, ulang):
    """AUC per horizon untuk kohort (boleh berulang) — grup = rekaman asli supaya duplikat tidak bocor."""
    a = []
    for h in v.HOR_TETAP:
        pos = u.ambil_scd(f, h)
        bag = [pos[pos.rec == rid].assign(rec=f'{rid}#{k}', grp=rid) for k, rid in enumerate(tetap_ids)]
        negg = neg if 'grp' in neg.columns else neg.assign(grp=neg.rec)
        d = pd.concat(bag + [negg])
        d['y'] = (d.db == 'sddb').astype(int)
        d['rec'] = d.db + ':' + d.rec
        X, y, g = d[CIRI].to_numpy(), d.y.to_numpy(), d.grp.to_numpy()
        ps = []
        for b in range(ulang):
            p = np.zeros(len(d))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
                p[te] = u.model(jenis).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            ps.append(p)
        s = pd.Series(np.mean(ps, axis=0)).groupby(d.rec.to_numpy()).mean()
        yy = d.groupby('rec').y.first().reindex(s.index)
        a.append(roc_auc_score(yy, s))
    return np.array(a)


def refit():
    f = fitur()
    r = np.random.default_rng(2)
    nsr, chf = kontrol(f, r)
    tetap = kohort_tetap(f)
    print(f'\n== Bootstrap dengan latih ulang, kohort tetap {len(tetap)} pasien: AUC(0) - AUC(360)', flush=True)
    for jenis, B, ulang in (('logistik', 200, 2), ('rf', 60, 1)):
        for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
            a0 = auc_horizon_refit(f, neg, tetap, jenis, 2)
            ds = []
            ctrl = neg.rec.unique()
            for _ in range(B):
                ids = list(r.choice(tetap, len(tetap)))
                cb = r.choice(ctrl, len(ctrl))
                nb = pd.concat([neg[neg.rec == c].assign(rec=f'{c}#{k}') for k, c in enumerate(cb)])
                nb_grp = nb.rec.str.split('#').str[0]
                a = auc_horizon_refit(f, nb.assign(grp=nb_grp), ids, jenis, ulang)
                ds.append(a[0] - a[-1])
            ds = np.array(ds)
            print(f'  {jenis:8s} lawan {nama}: AUC per horizon {np.round(a0, 3).tolist()}; selisih '
                  f'{a0[0] - a0[-1]:+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}] (B={B})',
                  flush=True)


# ---------------------------------------------------------------- daya
def suntik(f, delta, arah, r):
    f = f.copy()
    s = f.db == 'sddb'
    t = f.loc[s, 'menit_sebelum_vf'].to_numpy()
    efek = delta * np.exp(-t / 60.0)
    per = r.normal(1.0, 0.5, size=f.loc[s, 'rec'].nunique())
    kali = pd.Series(per, index=f.loc[s, 'rec'].unique()).reindex(f.loc[s, 'rec']).to_numpy()
    f.loc[s, CIRI] = f.loc[s, CIRI].to_numpy() + np.outer(efek * kali, arah)
    return f


def daya():
    f0 = fitur()
    sd = f0[CIRI].std().to_numpy()
    beda = (f0[f0.db == 'sddb'][CIRI].mean() - f0[f0.db == 'nsrdb'][CIRI].mean()).to_numpy()
    arah = beda / sd                                   # arah ke "lebih mirip sddb", satuan SD
    arah = arah / np.linalg.norm(arah) * sd
    print('\n== Simulasi daya: perubahan pra-VF buatan delta x SD, meluruh e^(-t/60), arah sddb - nsrdb', flush=True)
    for delta in (0.0, 0.5, 1.0, 2.0):
        tren_ok, dalam_ok, sel = 0, 0, []
        nsim = 20
        for k in range(nsim):
            r = np.random.default_rng(100 + k)
            f = suntik(f0, delta, arah, r)
            nsr, chf = kontrol(f, r)
            tetap = kohort_tetap(f)
            a = []
            for h in v.HOR_TETAP:
                pos = u.ambil_scd(f, h)
                sc, subj = v.skor_subjek(pos[pos.rec.isin(tetap)], nsr, 'semua', 'logistik')
                a.append((subj, sc))
            idx = a[0][1].index
            y = a[0][0].reindex(idx).to_numpy()
            S = np.column_stack([x[1].reindex(idx).to_numpy() for x in a])
            pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
            ds = []
            for _ in range(500):
                ii = np.r_[r.choice(pi, len(pi)), r.choice(ni, len(ni))]
                ds.append(roc_auc_score(y[ii], S[ii, 0]) - roc_auc_score(y[ii], S[ii, -1]))
            d0 = roc_auc_score(y, S[:, 0]) - roc_auc_score(y, S[:, -1])
            sel.append(d0)
            tren_ok += np.percentile(ds, 2.5) > 0
            # dalam-pasien (jauh >= 120), logistik, 3 ulangan
            s = f[f.db == 'sddb']
            pos = s[s.menit_sebelum_vf < 30]
            n = pos.groupby('rec').size()
            pos = pos[pos.rec.isin(n[n >= 3].index)]
            neg = s[(s.menit_sebelum_vf >= 120) & s.rec.isin(pos.rec)]
            neg = neg.sample(frac=1, random_state=k).groupby('rec').head(6)
            d = pd.concat([pos[pos.rec.isin(neg.rec)].assign(y=1), neg.assign(y=0)]).reset_index(drop=True)
            w.ULANG = 3
            _, lo, _ = w.boot(d, w.oof(d, 'logistik', CIRI), r)
            dalam_ok += lo > 0.5
        print(f'  delta {delta:.1f} SD: AUC(0)-AUC(360) rerata {np.mean(sel):+.3f}; uji horizon menangkap '
              f'{tren_ok}/{nsim}; uji dalam-pasien menangkap {dalam_ok}/{nsim}', flush=True)


# ---------------------------------------------------------------- halus
def halus():
    f = fitur()
    r = np.random.default_rng(3)
    nsr, chf = kontrol(f, r)
    print('\n== Pita 10 menit dalam 30 menit terakhir (subjek dengan >= 1 jendela di pita)', flush=True)
    for lo_, hi_ in ((0, 10), (10, 20), (20, 30), (30, 60), (120, 150), (360, 390)):
        g = f[(f.db == 'sddb') & (f.menit_sebelum_vf >= lo_) & (f.menit_sebelum_vf < hi_)]
        for jenis in ('logistik', 'rf'):
            out = []
            for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
                h = u.uji(f, g, neg, f'pita_{lo_}', 'semua', jenis, r)
                out.append(f'{nama} {h["auc"]:.3f} [{h["lo"]:.2f}, {h["hi"]:.2f}]')
            print(f'  {lo_:3d}-{hi_:3d} menit {jenis:8s} n={g.rec.nunique():2d}  ' + '   '.join(out), flush=True)
    print('\n== Dalam-pasien per pita 10 menit lawan jendela >= 120 menit, pasien yang sama', flush=True)
    s = f[f.db == 'sddb']
    for lo_, hi_ in ((0, 10), (10, 20), (20, 30)):
        pos = s[(s.menit_sebelum_vf >= lo_) & (s.menit_sebelum_vf < hi_)]
        neg = s[(s.menit_sebelum_vf >= 120) & s.rec.isin(pos.rec)]
        neg = neg.sample(frac=1, random_state=7).groupby('rec').head(6)
        d = pd.concat([pos[pos.rec.isin(neg.rec)].assign(y=1), neg.assign(y=0)]).reset_index(drop=True)
        for jenis in ('logistik', 'rf'):
            w.ULANG = 10
            a, lo, hi = w.boot(d, w.oof(d, jenis, CIRI), r)
            print(f'  {lo_:2d}-{hi_:2d} menit {jenis:8s} AUC {a:.3f} [{lo:.2f}, {hi:.2f}]  pasien {d.rec.nunique()}, '
                  f'jendela {int(d.y.sum())}/{int((1 - d.y).sum())}', flush=True)


# ---------------------------------------------------------------- sinus
def dalam(f, r, jauh=120):
    s = f[f.db == 'sddb']
    pos = s[s.menit_sebelum_vf < 30]
    n = pos.groupby('rec').size()
    pos = pos[pos.rec.isin(n[n >= 3].index)]
    neg = s[(s.menit_sebelum_vf >= jauh) & s.rec.isin(pos.rec)]
    neg = neg.sample(frac=1, random_state=11).groupby('rec').head(6)
    return pd.concat([pos[pos.rec.isin(neg.rec)].assign(y=1), neg.assign(y=0)]).reset_index(drop=True)


def sinus():
    for label, saring in (('tanpa VT', ['vt']), ('sinus saja (tanpa AF/flutter, pacu, VT)', ['af', 'pacu', 'vt'])):
        f = fitur(saring)
        r = np.random.default_rng(4)
        nsr, chf = kontrol(f, r)
        print(f'\n== {label}: jendela tersisa sddb {int((f.db == "sddb").sum())}, nsrdb {int((f.db == "nsrdb").sum())}, '
              f'chfdb {int((f.db == "chfdb").sum())}', flush=True)
        for jenis in ('logistik', 'rf'):
            for h in (0, 60, 360):
                g = u.ambil_scd(f, h)
                a = u.uji(f, g, nsr, 'h', 'semua', jenis, r)
                b = u.uji(f, g, chf, 'h', 'semua', jenis, r)
                print(f'  {jenis:8s} h={h:3d} n={a["n_pos"]:2d}  lawan nsrdb {a["auc"]:.3f} [{a["lo"]:.2f}, {a["hi"]:.2f}]'
                      f'   lawan chfdb {b["auc"]:.3f} [{b["lo"]:.2f}, {b["hi"]:.2f}]', flush=True)
            for cs in ('semua', 'tanpa_ektopik'):
                d = dalam(f, r)
                w.ULANG = 10
                a, lo, hi = w.boot(d, w.oof(d, jenis, u.SET[cs]), r)
                print(f'  {jenis:8s} dalam-pasien (>= 120 menit) {cs:13s} AUC {a:.3f} [{lo:.2f}, {hi:.2f}]  '
                      f'pasien {d.rec.nunique()}', flush=True)


# ---------------------------------------------------------------- pasangan
def pasangan():
    f = fitur()
    r = np.random.default_rng(5)
    print('\n== Dalam-pasien berpasangan: per pasien rerata skor OOF dekat (0-30) minus jauh (>= 120)', flush=True)
    d = dalam(f, r)
    for jenis in ('logistik', 'rf'):
        for cs in ('semua', 'tanpa_ektopik'):
            w.ULANG = 10
            p = w.oof(d, jenis, u.SET[cs])
            t = d.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
            sel = (t[1] - t[0]).dropna()
            auc_p = [roc_auc_score(g.y, g.p) for _, g in d.assign(p=p).groupby('rec') if g.y.nunique() == 2]
            bt = binomtest(int((sel > 0).sum()), len(sel))
            print(f'  {jenis:8s} {cs:13s} pasien {len(sel)}: dekat > jauh pada {int((sel > 0).sum())}; median selisih '
                  f'{sel.median():+.3f}; uji tanda p {bt.pvalue:.3f}; Wilcoxon p {wilcoxon(sel).pvalue:.3f}; '
                  f'median AUC per pasien {np.median(auc_p):.3f}', flush=True)


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['irama', 'palsu', 'halus', 'sinus', 'pasangan', 'refit', 'daya']
    for b in pilih:
        globals()[b]()

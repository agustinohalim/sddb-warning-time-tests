"""Artikel EKG 1 — paket B sesudah telaah panel 25 Oktober 2026 (Telaah/2026-10-25-EKG1/report.md).

Isu yang dijawab (id temuan panel):
  M1, X9, DA-X3  simulasi daya memakai statistik dalam-pasien yang berbeda dari data nyata (rerata
                 selisih skor berpasangan, bukan AUC tingkat jendela dengan uji permutasi).
  M2             ambang disesuaikan ukuran dari 100 simulasi nol, dan ukuran diperiksa pada simulasi yang sama.
  DA-X2          daya uji horizon rendah sebagian karena langit-langit AUC lawan nsrdb; ulang lawan chfdb.
  M3, M4, M6     uji permutasi mengabaikan autokorelasi; kontrol pergeseran memakai pasien lain dan
                 jumlah ulangan CV berbeda dari kontras yang diamati.
  M5             selang RF kohort tetap (+0,002 batas bawah) dari 60 undian saja.
  M7, DA-X5      tanpa uji primer, tanpa koreksi kelipatan.

Bagian (argumen; tanpa argumen semua, urut):
  daya      simulasi daya kohort bootstrap seperti 11/12 (12 pasien kohort tetap untuk kedua uji, nsrdb dan
            chfdb diambil ulang), regresi logistik. Statistik per simulasi:
              d_auc   AUC tingkat jendela dalam-pasien (skor OOF, 3 ulangan CV) — statistik data nyata
              d_sel   rerata selisih skor dekat - jauh per pasien (statistik Tabel 4 lama)
              h_nsr   AUC(0) - AUC(6 jam) lawan nsrdb, tingkat pasien
              h_chf   sama, lawan chfdb (tidak di langit-langit)
              hp_nsr  horizon berpasangan: rerata per pasien sddb skor(0) - skor(6 jam), lawan nsrdb
            Nol: 1.000 simulasi (benih 5000 + k) untuk ambang, 500 simulasi segar (benih 9000 + k) untuk
            memeriksa ukuran. Tiap sel perubahan 200 simulasi (benih 5000 + k, berpasangan dengan nol k).
            Hanya horizon 0 dan 360 menit yang dihitung; keduanya satu-satunya yang dipakai statistik.
  permsim   uji permutasi dalam-pasien DI DALAM simulasi: label dekat/jauh diacak dalam tiap pasien, model
            dilatih ulang, 99 permutasi, p = (1 + #perm >= asli) / 100, tolak bila p <= 0,05. Nol 200
            simulasi segar (benih 9000 + k), searah_k 1 dan 2 SD dan denyut +20 masing-masing 100.
  geser     kontrol pergeseran sepadan pada data nyata, PASIEN SAMA: hanya pasien yang rekamannya cukup
            panjang untuk geometri pergeseran (jendela >= 240 menit sebelum VF). Kontras teramati (0-30
            menit lawan >= 2 jam) dan 200 undian pergeseran dihitung pada pasien itu dengan 10 ulangan CV
            yang sama; juga simpangan baku AUC antarulangan CV. Uji primer: random forest, AUC teramati
            lawan sebaran pergeseran, p = (1 + #undian >= teramati) / 201.
  refit     bootstrap dengan latih ulang untuk AUC(0) - AUC(6 jam) random forest lawan nsrdb, kohort tetap,
            500 undian, 2 ulangan CV.
  holm      p terkoreksi Holm dan Benjamini-Hochberg untuk keluarga uji dalam-pasien (Tabel 6 dan geser).
Keluaran: keluaran/b_daya.csv, b_permsim.csv, b_geser_<model>.csv, b_refit.csv, b_*.log (lewat >)
Pakai:    PYTHONPATH=. python -u 14_paket_b.py [bagian ...] > keluaran/b.log
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
w = import_module('07_dalam_pasien')
x = import_module('09_revisi_panel')
y10 = import_module('10_daya_ulang')
b11 = import_module('11_daya_bootstrap')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
H0, H6 = 0, 360
ULANG_H, ULANG_D = 5, 3
N_NOL, N_SEGAR, N_SEL = 1000, 500, 200
SEL = [('searah_k', 1.0), ('searah_k', 2.0), ('tegak_k', 1.0), ('nadi', 10), ('nadi', 20),
       ('ektopi', 2.0), ('ektopi', 4.0), ('ektopi', 0.5), ('ektopi', 0.25)]
JOBS = 20


# ---------------------------------------------------------------- statistik pada kohort bootstrap
def cv_skor(d, ciri, ulang, jenis='logistik'):
    X, yy, g = d[ciri].to_numpy(), d.y.to_numpy(), d.grp.to_numpy()
    k = min(5, len(np.unique(g)))            # undian langka dengan < 5 pasien berbeda
    ps = []
    for b in range(ulang):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=b).split(X, yy, g):
            p[te] = u.model(jenis).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def kontrol_boot(f, db, ids, r):
    g = f[f.db == db]
    return pd.concat([g[g.rec == rid].sample(n=min(6, (g.rec == rid).sum()), random_state=int(r.integers(1 << 30)))
                      .assign(rec=f'c{rid}#{k}', grp=f'c{rid}') for k, rid in enumerate(ids)]).assign(y=0)


def horizon_stat(f, pas, neg):
    """(AUC(0) - AUC(6 jam), rerata per pasien skor(0) - skor(6 jam)) untuk satu kelompok kontrol."""
    S, yy = [], None
    for h in (H0, H6):
        pos = u.ambil_scd(f, h)
        bag = pd.concat([pos[pos.rec == rid].assign(rec=f's{rid}#{k}', grp=f's{rid}')
                         for k, rid in enumerate(pas)]).assign(y=1)
        d = pd.concat([bag, neg]).reset_index(drop=True)
        sc = pd.Series(cv_skor(d, CIRI, ULANG_H)).groupby(d.rec.to_numpy()).mean()
        if yy is None:
            yy = d.groupby('rec').y.first().reindex(sc.index)
        S.append(sc.reindex(yy.index).to_numpy())
    yv = yy.to_numpy()
    pos = yv == 1
    return roc_auc_score(yv, S[0]) - roc_auc_score(yv, S[1]), float(np.mean(S[0][pos] - S[1][pos]))


def data_dalam(f, pas):
    dd = x.dalam(f, None)
    return pd.concat([dd[dd.rec == rid].assign(rec=f'{rid}#{k}', grp=rid)
                      for k, rid in enumerate(pas)]).reset_index(drop=True)


def dalam_stat(d, ulang=ULANG_D):
    p = cv_skor(d, CIRI, ulang)
    t = d.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
    return roc_auc_score(d.y, p), (t[1] - t[0]).dropna().mean()


def satu_sim(f0, arah, tetap, nsr_ids, chf_ids, jenis, besar, benih):
    r = np.random.default_rng(benih)
    pas = list(r.choice(tetap, len(tetap)))
    nsr = list(r.choice(nsr_ids, len(nsr_ids)))
    chf = list(r.choice(chf_ids, len(chf_ids)))
    f = b11.buat_data(f0, arah, 'nol' if jenis == 'segar' else jenis, besar, r)
    h_nsr, hp_nsr = horizon_stat(f, pas, kontrol_boot(f, 'nsrdb', nsr, r))
    h_chf, _ = horizon_stat(f, pas, kontrol_boot(f, 'chfdb', chf, r))
    d_auc, d_sel = dalam_stat(data_dalam(f, pas))
    return dict(jenis=jenis, besar=besar, benih=benih, d_auc=d_auc, d_sel=d_sel, h_nsr=h_nsr, h_chf=h_chf,
                hp_nsr=hp_nsr)


def siapkan_sim():
    f0 = x.fitur()
    return (f0, y10.arah_dasar(f0), x.kohort_tetap(f0), sorted(f0[f0.db == 'nsrdb'].rec.unique()),
            sorted(f0[f0.db == 'chfdb'].rec.unique()))


def jalankan(path, tugas, fungsi, *arg):
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['jenis', 'besar', 'benih'])
    kunci = set(zip(sudah.jenis, sudah.besar.astype(float), sudah.benih))
    sisa = [t for t in tugas if (t[0], float(t[1]), t[2]) not in kunci]
    for i in range(0, len(sisa), 200):
        t0 = time.time()
        hasil = Parallel(n_jobs=JOBS)(delayed(fungsi)(*arg, j, b, s) for j, b, s in sisa[i:i + 200])
        pd.DataFrame(hasil).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        print(f'  {i + len(hasil)}/{len(sisa)} simulasi, {time.time() - t0:.0f} dtk', flush=True)


# ---------------------------------------------------------------- daya
def daya():
    f0, arah, tetap, nsr_ids, chf_ids = siapkan_sim()
    print(f'== Simulasi daya, statistik AUC jendela: {len(tetap)} pasien kohort tetap, nsrdb {len(nsr_ids)}, '
          f'chfdb {len(chf_ids)}', flush=True)
    tugas = ([('nol', 0, 5000 + k) for k in range(N_NOL)] + [('segar', 0, 9000 + k) for k in range(N_SEGAR)]
             + [(j, b, 5000 + k) for j, b in SEL for k in range(N_SEL)])
    jalankan(os.path.join(OUT, 'b_daya.csv'), tugas, satu_sim, f0, arah, tetap, nsr_ids, chf_ids)
    ringkas_daya()


def ambang(nol, kol, sisi):
    if sisi == 1:
        return -np.inf, nol[kol].quantile(0.95)
    return tuple(nol[kol].quantile([0.025, 0.975]))


def laju(g, kol, lo, hi):
    return float(((g[kol] < lo) | (g[kol] > hi)).mean())


def ringkas_daya():
    h = pd.read_csv(os.path.join(OUT, 'b_daya.csv'))
    nol, segar = h[h.jenis == 'nol'], h[h.jenis == 'segar']
    uji = [('h_nsr', 2, 'horizon lawan nsrdb'), ('hp_nsr', 2, 'horizon berpasangan lawan nsrdb'),
           ('h_chf', 2, 'horizon lawan chfdb'), ('d_sel', 2, 'dalam-pasien rerata selisih'),
           ('d_auc', 2, 'dalam-pasien AUC jendela, dua sisi'), ('d_auc', 1, 'dalam-pasien AUC jendela, satu sisi')]
    print(f'\n== Ringkasan daya: ambang dari {len(nol)} simulasi nol, ukuran diperiksa pada {len(segar)} simulasi '
          f'nol segar; selang 95 % dari bootstrap simulasi nol dan simulasi sel bersama (1.000 undian)')
    r = np.random.default_rng(31)
    nol_v = nol.reset_index(drop=True)
    for kol, sisi, nama in uji:
        lo, hi = ambang(nol, kol, sisi)
        print(f'\n  {nama} [{kol}]: nol rerata {nol[kol].mean():+.3f} sd {nol[kol].std():.3f}; ambang '
              f'({lo:+.3f}, {hi:+.3f}); ukuran pada nol segar {laju(segar, kol, lo, hi):.3f}', flush=True)
        for j, b in SEL:
            g = h[(h.jenis == j) & (h.besar == b)].reset_index(drop=True)
            if not len(g):
                continue
            bs = []
            for _ in range(1000):
                nb = nol_v.iloc[r.integers(0, len(nol_v), len(nol_v))]
                gb = g.iloc[r.integers(0, len(g), len(g))]
                bs.append(laju(gb, kol, *ambang(nb, kol, sisi)))
            print(f'    {j:8s} {b:5g} n={len(g):3d}: rerata {g[kol].mean():+.3f}, daya {laju(g, kol, lo, hi):.2f} '
                  f'[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]', flush=True)
    # rasio dalam-pasien (AUC jendela, dua sisi) terhadap horizon lawan nsrdb dan chfdb, berpasangan per benih
    print('\n  selisih dan rasio daya, AUC jendela dalam-pasien dua sisi terhadap horizon (bootstrap berpasangan atas '
          'simulasi, ambang ikut diundi):')
    for hk in ('h_nsr', 'h_chf'):
        for j, b in SEL:
            g = h[(h.jenis == j) & (h.besar == b)].reset_index(drop=True)
            if not len(g):
                continue

            def dua(nb, gb):
                dl, dh = ambang(nb, 'd_auc', 2)
                hl, hh = ambang(nb, hk, 2)
                a = ((gb.d_auc < dl) | (gb.d_auc > dh)).mean()
                c = ((gb[hk] < hl) | (gb[hk] > hh)).mean()
                return a - c, a / c if c > 0 else np.nan
            s0, q0 = dua(nol_v, g)
            bs = [dua(nol_v.iloc[r.integers(0, len(nol_v), len(nol_v))], g.iloc[r.integers(0, len(g), len(g))])
                  for _ in range(1000)]
            bs = np.array(bs)
            print(f'    {hk} {j:8s} {b:5g}: selisih {s0:+.2f} [{np.nanpercentile(bs[:, 0], 2.5):+.2f}, '
                  f'{np.nanpercentile(bs[:, 0], 97.5):+.2f}], rasio {q0:.1f} [{np.nanpercentile(bs[:, 1], 2.5):.1f}, '
                  f'{np.nanpercentile(bs[:, 1], 97.5):.1f}]', flush=True)


# ---------------------------------------------------------------- permsim
def satu_perm(f0, arah, tetap, nsr_ids, chf_ids, jenis, besar, benih, n_perm=99):
    r = np.random.default_rng(benih)
    pas = list(r.choice(tetap, len(tetap)))
    r.choice(nsr_ids, len(nsr_ids))          # menjaga urutan undian seperti satu_sim
    r.choice(chf_ids, len(chf_ids))
    f = b11.buat_data(f0, arah, 'nol' if jenis == 'segar' else jenis, besar, r)
    d = data_dalam(f, pas)
    a0, _ = dalam_stat(d)
    rp = np.random.default_rng(benih + 1)
    ap = []
    for _ in range(n_perm):
        dp = d.copy()
        dp['y'] = dp.groupby('rec').y.transform(lambda s_: rp.permutation(s_.to_numpy()))
        ap.append(dalam_stat(dp)[0])
    p = (1 + (np.array(ap) >= a0).sum()) / (n_perm + 1)
    return dict(jenis=jenis, besar=besar, benih=benih, d_auc=a0, p=p)


def permsim():
    f0, arah, tetap, nsr_ids, chf_ids = siapkan_sim()
    print('\n== Uji permutasi dalam-pasien di dalam simulasi (99 permutasi, latih ulang)', flush=True)
    tugas = ([('segar', 0, 9000 + k) for k in range(200)]
             + [(j, b, 5000 + k) for j, b in (('searah_k', 1.0), ('searah_k', 2.0), ('nadi', 20)) for k in range(100)])
    path = os.path.join(OUT, 'b_permsim.csv')
    jalankan(path, tugas, satu_perm, f0, arah, tetap, nsr_ids, chf_ids)
    h = pd.read_csv(path)
    for (j, b), g in h.groupby(['jenis', 'besar'], sort=False):
        print(f'  {j:8s} {b:5g} n={len(g):3d}: tolak p <= 0,05 pada {(g.p <= 0.05).mean():.2f}; AUC rerata '
              f'{g.d_auc.mean():.3f}', flush=True)


# ---------------------------------------------------------------- geser (pasien sama, ulangan sama)
def oof_ulang(d, jenis, ulang):
    X, y, g = d[CIRI].to_numpy(), d.y.to_numpy(), d.rec.to_numpy()
    ps = []
    for b in range(ulang):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
            p[te] = u.model(jenis).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.array(ps)


def geser():
    f = x.fitur()
    s = f[f.db == 'sddb']
    utama = x.dalam(f, None)
    panjang = s.groupby('rec').menit_sebelum_vf.max()
    ids = sorted(set(utama.rec) & set(panjang[panjang >= 240].index))
    d_obs = utama[utama.rec.isin(ids)].reset_index(drop=True)
    print(f'\n== Kontrol pergeseran, pasien sama ({len(ids)} dari {utama.rec.nunique()}), 10 ulangan CV untuk '
          f'kontras teramati dan tiap undian', flush=True)
    for jenis in ('logistik', 'rf'):
        t0 = time.time()
        P = oof_ulang(d_obs, jenis, 10)
        a_ul = np.array([roc_auc_score(d_obs.y, p) for p in P])
        a_obs = roc_auc_score(d_obs.y, P.mean(axis=0))
        r = np.random.default_rng(23)
        a, n_pas = [], []
        for k in range(200):
            bag = []
            for rid in ids:
                g = s[s.rec == rid]
                m = r.uniform(120, g.menit_sebelum_vf.max() - 120)
                dk = g[(g.menit_sebelum_vf >= m) & (g.menit_sebelum_vf < m + 30)]
                jh = g[g.menit_sebelum_vf >= m + 120]
                if len(dk) < 3 or len(jh) == 0:
                    continue
                jh = jh.sample(n=min(6, len(jh)), random_state=int(r.integers(1 << 30)))
                bag += [dk.assign(y=1), jh.assign(y=0)]
            d = pd.concat(bag).reset_index(drop=True)
            a.append(roc_auc_score(d.y, oof_ulang(d, jenis, 10).mean(axis=0)))
            n_pas.append(d.rec.nunique())
            if (k + 1) % 50 == 0:
                print(f'    ... {jenis} {k + 1} undian, {time.time() - t0:.0f} dtk', flush=True)
        a = np.array(a)
        p = (1 + (a >= a_obs).sum()) / (len(a) + 1)
        pd.DataFrame({'auc': a, 'n_pasien': n_pas}).to_csv(os.path.join(OUT, f'b_geser_{jenis}.csv'), index=False)
        print(f'  {jenis:8s}: teramati {a_obs:.3f} (sd antarulangan CV {a_ul.std(ddof=1):.3f}, galat baku Monte '
              f'Carlo {a_ul.std(ddof=1) / np.sqrt(len(a_ul)):.3f}); pergeseran rerata {a.mean():.3f} '
              f'[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}], pasien per undian '
              f'{min(n_pas)}-{max(n_pas)}; p {p:.3f}; {time.time() - t0:.0f} dtk', flush=True)


# ---------------------------------------------------------------- refit (RF, 500 undian)
def satu_refit(f, neg, tetap, benih):
    r = np.random.default_rng(benih)
    ids = list(r.choice(tetap, len(tetap)))
    ctrl = neg.rec.unique()
    cb = r.choice(ctrl, len(ctrl))
    nb = pd.concat([neg[neg.rec == c].assign(rec=f'{c}#{k}', grp=c) for k, c in enumerate(cb)])
    a = []
    for h in (H0, H6):
        pos = u.ambil_scd(f, h)
        bag = [pos[pos.rec == rid].assign(rec=f'{rid}#{k}', grp=rid) for k, rid in enumerate(ids)]
        d = pd.concat(bag + [nb])
        d['y'] = (d.db == 'sddb').astype(int)
        d['rec'] = d.db + ':' + d.rec
        d = d.reset_index(drop=True)
        X, y, g = d[CIRI].to_numpy(), d.y.to_numpy(), d.grp.to_numpy()
        ps = []
        for b in range(2):
            p = np.zeros(len(d))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
                p[te] = u.model('rf').set_params(n_jobs=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            ps.append(p)
        sc = pd.Series(np.mean(ps, axis=0)).groupby(d.rec.to_numpy()).mean()
        yy = d.groupby('rec').y.first().reindex(sc.index)
        a.append(roc_auc_score(yy, sc))
    return a[0] - a[1]


def refit():
    f = x.fitur()
    nsr, _ = x.kontrol(f, np.random.default_rng(2))     # kontrol sama dengan 09_revisi_panel.refit
    tetap = x.kohort_tetap(f)
    t0 = time.time()
    ds = np.array(Parallel(n_jobs=JOBS)(delayed(satu_refit)(f, nsr, tetap, 600 + k) for k in range(500)))
    pd.DataFrame({'sel': ds}).to_csv(os.path.join(OUT, 'b_refit.csv'), index=False)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # galat Monte Carlo batas persentil: bootstrap atas 500 undian
    r = np.random.default_rng(3)
    blo = [np.percentile(r.choice(ds, len(ds)), 2.5) for _ in range(2000)]
    print(f'\n== RF lawan nsrdb, kohort tetap {len(tetap)} pasien, 500 undian latih ulang: AUC(0) - AUC(6 jam) '
          f'selang [{lo:+.3f}, {hi:+.3f}]; galat baku batas bawah {np.std(blo):.3f}; bagian undian <= 0 '
          f'{(ds <= 0).mean():.3f}; {time.time() - t0:.0f} dtk', flush=True)


# ---------------------------------------------------------------- holm
def koreksi(pv):
    """(Holm, Benjamini-Hochberg) untuk larik p."""
    pv = np.asarray(pv, float)
    m, o = len(pv), np.argsort(pv)
    h = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * pv[o]))
    b = np.minimum(1, np.minimum.accumulate((m / np.arange(m, 0, -1)) * pv[o][::-1])[::-1])
    hh, bb = np.empty(m), np.empty(m)
    hh[o], bb[o] = h, b
    return hh, bb


def holm():
    """Keluarga uji dalam-pasien eksploratif (keluaran/b_keluarga_p.csv: p dari Tabel 6 naskah dengan baris log
    sumbernya). Uji primer (bagian geser, random forest) tidak masuk keluarga: ia diuji sendiri pada 0,05."""
    p = pd.read_csv(os.path.join(OUT, 'b_keluarga_p.csv'))
    p['holm'], p['bh'] = koreksi(p.p)
    print('\n== Koreksi kelipatan, keluarga uji dalam-pasien')
    print(p.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['daya', 'permsim', 'geser', 'refit', 'holm']
    for bag in pilih:
        globals()[bag]()

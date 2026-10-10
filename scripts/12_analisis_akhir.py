"""Artikel EKG 1 — analisis sesudah telaah panel kelima 11 Oktober 2026 (Telaah/2026-10-11-EKG1/report.md).

Bagian (argumen; tanpa argumen semua, urut):
  dalam12   uji dalam-pasien pada simulasi kohort bootstrap dengan 12 pasien kohort tetap yang sama
            dengan uji horizon dan ambang dua sisi (persentil 2,5/97,5 dari 100 simulasi nol), supaya
            perbandingan dengan uji horizon setara (jumlah pasien dan sisi sama). Logistik, 100 simulasi
            per sel, benih sama dengan 11_daya_bootstrap.py. Hasil horizon dibaca dari daya_bootstrap.csv.
  rf        simulasi daya kohort bootstrap dengan random forest (2 ulangan CV): nol 40 simulasi, searah_k
            1 dan 2 SD, denyut +20, ektopi x4 masing-masing 20; horizon (12 pasien, semua ciri) dan
            dalam-pasien (12 pasien), ambang dua sisi dari simulasi nol.
  geser     kontrol pergeseran waktu yang sepadan dengan kontras utama: blok "dekat palsu" = jendela
            yang berakhir m sampai m + 30 menit sebelum VF (m >= 120, acak per pasien), jendela jauh =
            berakhir >= m + 120 menit (jarak 90 menit seperti kontras utama), sampai 6; semua pasien
            yang memenuhi; 200 undian per model.
  hz128     cek kuantisasi 128 Hz: waktu denyut sddb dan chfdb dibulatkan ke kisi 1/128 detik, ciri dihitung
            ulang, lalu uji horizon (semua pasien per horizon, lawan nsrdb dan chfdb) dan kontras
            dalam-pasien diulang dan dibandingkan dengan ciri asli (5 ulangan CV).
  perpasien rincian per pasien kontras dalam-pasien RF (10 ulangan CV): selisih skor dekat - jauh, jendela
            VT dan rerata frac_ektopik dekat dan jauh; AUC tinggalkan-satu-pasien.
Keluaran: keluaran/akhir_*.csv, keluaran/akhir.log (lewat >)
Pakai:    python -u 12_analisis_akhir.py [bagian ...] > keluaran/akhir.log
"""
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u = import_module('03_uji')
v = import_module('04_uji_lanjut')
w = import_module('07_dalam_pasien')
x = import_module('09_revisi_panel')
y10 = import_module('10_daya_ulang')
b11 = import_module('11_daya_bootstrap')
fit = import_module('02_fitur')
OUT = u.OUT
CIRI = u.CIRI_SEMUA
HOR = v.HOR_TETAP


def cv_skor(d, ciri, ulang, jenis):
    X, yy, g = d[ciri].to_numpy(), d.y.to_numpy(), d.grp.to_numpy()
    ps = []
    for b in range(ulang):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, yy, g):
            p[te] = u.model(jenis).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def dalam_stat(f, pas, r, jenis='logistik', ulang=3):
    dd = x.dalam(f, r)
    d = pd.concat([dd[dd.rec == rid].assign(rec=f'{rid}#{k}', grp=rid) for k, rid in enumerate(pas)])
    d = d.reset_index(drop=True)
    p = cv_skor(d, CIRI, ulang, jenis)
    t = d.assign(p=p).groupby(['rec', 'y']).p.mean().unstack()
    return (t[1] - t[0]).dropna().mean()


def horizon_stat(f, pas, kon, r, jenis, ulang):
    nsr = f[f.db == 'nsrdb']
    neg = pd.concat([nsr[nsr.rec == rid].sample(n=6, random_state=int(r.integers(1 << 30)))
                     .assign(rec=f'n{rid}#{k}', grp=f'n{rid}') for k, rid in enumerate(kon)]).assign(y=0)
    S, yy = [], None
    for h in HOR:
        pos = u.ambil_scd(f, h)
        bag = pd.concat([pos[pos.rec == rid].assign(rec=f's{rid}#{k}', grp=f's{rid}')
                         for k, rid in enumerate(pas)]).assign(y=1)
        d = pd.concat([bag, neg]).reset_index(drop=True)
        sc = pd.Series(cv_skor(d, CIRI, ulang, jenis)).groupby(d.rec.to_numpy()).mean()
        if yy is None:
            yy = d.groupby('rec').y.first().reindex(sc.index)
        S.append(sc.reindex(yy.index).to_numpy())
    yv = yy.to_numpy()
    return roc_auc_score(yv, S[0]) - roc_auc_score(yv, S[-1])


SEL = [('nol', 0), ('searah_k', 1.0), ('searah_k', 2.0), ('tegak_k', 1.0), ('nadi', 10), ('nadi', 20),
       ('ektopi', 2.0), ('ektopi', 4.0), ('ektopi', 0.5), ('ektopi', 0.25)]


def daya_dua_sisi(h, kol, nol_mask):
    lo, hi = h.loc[nol_mask, kol].quantile([0.025, 0.975])
    return lambda g: ((g[kol] < lo) | (g[kol] > hi)).mean()


# ---------------------------------------------------------------- dalam12
def dalam12():
    f0 = x.fitur()
    arah = y10.arah_dasar(f0)
    tetap = x.kohort_tetap(f0)
    kon_ids = sorted(f0[f0.db == 'nsrdb'].rec.unique())
    path = os.path.join(OUT, 'akhir_dalam12.csv')
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    print(f'== Dalam-pasien pada 12 pasien kohort tetap, dua sisi: {tetap}', flush=True)
    for jenis, besar in SEL:
        t0 = time.time()
        for k in range(100):
            if len(sudah) and ((sudah.jenis == jenis) & (sudah.besar == besar) & (sudah.sim == k)).any():
                continue
            r = np.random.default_rng(5000 + k)
            pas = list(r.choice(tetap, len(tetap)))
            r.choice(kon_ids, len(kon_ids))          # menjaga urutan undian seperti 11
            r.choice(tetap, len(tetap))
            f = b11.buat_data(f0, arah, jenis, besar, r)
            ds = dalam_stat(f, pas, r)
            pd.DataFrame([dict(jenis=jenis, besar=besar, sim=k, d12=ds)]).to_csv(
                path, mode='a', header=not os.path.exists(path), index=False)
        print(f'  {jenis:8s} {besar:5g}: {time.time() - t0:.0f} dtk', flush=True)
    d = pd.read_csv(path)
    hb = pd.read_csv(os.path.join(OUT, 'daya_bootstrap.csv'))
    fd = daya_dua_sisi(d, 'd12', d.jenis == 'nol')
    fh = daya_dua_sisi(hb, 'h_sel', hb.jenis == 'nol')
    fd19 = daya_dua_sisi(hb, 'd_sel', hb.jenis == 'nol')
    print('\n  sel                horizon-12 dua sisi | dalam-pasien-12 dua sisi | dalam-pasien-19 dua sisi')
    for j, b in SEL:
        print(f'  {j:8s} {b:5g}:   {fh(hb[(hb.jenis == j) & (hb.besar == b)]):.2f}   |   '
              f'{fd(d[(d.jenis == j) & (d.besar == b)]):.2f}   |   {fd19(hb[(hb.jenis == j) & (hb.besar == b)]):.2f}',
              flush=True)


# ---------------------------------------------------------------- rf
def rf():
    f0 = x.fitur()
    arah = y10.arah_dasar(f0)
    tetap = x.kohort_tetap(f0)
    kon_ids = sorted(f0[f0.db == 'nsrdb'].rec.unique())
    path = os.path.join(OUT, 'akhir_rf.csv')
    sudah = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    sel = [('nol', 0, 40), ('searah_k', 1.0, 20), ('searah_k', 2.0, 20), ('nadi', 20, 20), ('ektopi', 4.0, 20)]
    print('\n== Simulasi daya kohort bootstrap, random forest (12 pasien, dua sisi)', flush=True)
    for jenis, besar, n in sel:
        t0 = time.time()
        for k in range(n):
            if len(sudah) and ((sudah.jenis == jenis) & (sudah.besar == besar) & (sudah.sim == k)).any():
                continue
            r = np.random.default_rng(7000 + k)
            pas = list(r.choice(tetap, len(tetap)))
            kon = list(r.choice(kon_ids, len(kon_ids)))
            f = b11.buat_data(f0, arah, jenis, besar, r)
            hs = horizon_stat(f, pas, kon, r, 'rf', 2)
            ds = dalam_stat(f, pas, r, 'rf', 2)
            pd.DataFrame([dict(jenis=jenis, besar=besar, sim=k, h=hs, d=ds)]).to_csv(
                path, mode='a', header=not os.path.exists(path), index=False)
        print(f'  {jenis:8s} {besar:5g}: {time.time() - t0:.0f} dtk', flush=True)
    d = pd.read_csv(path)
    fh = daya_dua_sisi(d, 'h', d.jenis == 'nol')
    fd = daya_dua_sisi(d, 'd', d.jenis == 'nol')
    n0 = d[d.jenis == 'nol']
    print(f'  nol: horizon rerata {n0.h.mean():+.3f} (sd {n0.h.std():.3f}); dalam-pasien rerata {n0.d.mean():+.3f} '
          f'(sd {n0.d.std():.3f})')
    for (j, b), g in d.groupby(['jenis', 'besar'], sort=False):
        print(f'  {j:8s} {b:5g} n={len(g):2d}: horizon {g.h.mean():+.3f} daya {fh(g):.2f} | dalam-pasien '
              f'{g.d.mean():+.3f} daya {fd(g):.2f}', flush=True)


# ---------------------------------------------------------------- geser
def geser():
    f = x.fitur()
    s = f[f.db == 'sddb']
    print('\n== Kontrol pergeseran waktu sepadan: blok 30 menit (m >= 120) lawan jendela >= m + 120 menit',
          flush=True)
    hasil = []
    for jenis, ulang in (('logistik', 3), ('rf', 2)):
        w.ULANG = ulang
        r = np.random.default_rng(23)
        a = []
        t0 = time.time()
        for k in range(200):
            bag = []
            for rid, g in s.groupby('rec'):
                mmax = g.menit_sebelum_vf.max()
                if mmax < 120 + 120:
                    continue
                m = r.uniform(120, mmax - 120)
                dk = g[(g.menit_sebelum_vf >= m) & (g.menit_sebelum_vf < m + 30)]
                jh = g[g.menit_sebelum_vf >= m + 120]
                if len(dk) < 3 or len(jh) == 0:
                    continue
                jh = jh.sample(n=min(6, len(jh)), random_state=int(r.integers(1 << 30)))
                bag += [dk.assign(y=1), jh.assign(y=0)]
            d = pd.concat(bag).reset_index(drop=True)
            a.append(roc_auc_score(d.y, w.oof(d, jenis, CIRI)))
        a = np.array(a)
        hasil.append(dict(model=jenis, rerata=a.mean(), lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5),
                          p95=np.percentile(a, 95), maks=a.max(), n_pasien=d.rec.nunique()))
        print(f'  {jenis:8s} pasien ~{d.rec.nunique()}: AUC rerata {a.mean():.3f} [{np.percentile(a, 2.5):.3f}, '
              f'{np.percentile(a, 97.5):.3f}], persentil 95 {np.percentile(a, 95):.3f}, maks {a.max():.3f}; '
              f'{time.time() - t0:.0f} dtk', flush=True)
        pd.DataFrame({'model': jenis, 'auc': a}).to_csv(os.path.join(OUT, f'akhir_geser_{jenis}.csv'), index=False)


# ---------------------------------------------------------------- hz128
def fitur_128():
    path = os.path.join(OUT, 'akhir_fitur_128.csv')
    if os.path.exists(path):
        return pd.read_csv(path, dtype={'rec': str})
    den = pd.read_csv(os.path.join(OUT, 'denyut.csv'), dtype={'rec': str})
    rek = pd.read_csv(os.path.join(OUT, 'rekaman.csv'), dtype={'rec': str})
    baris = []
    for _, r in rek.iterrows():
        if r.db not in ('sddb', 'chfdb', 'nsrdb'):
            continue
        g = den[(den.db == r.db) & (den.rec == r.rec)]
        t, sym = g.t.to_numpy(), g.sym.to_numpy().astype(str)
        if r.db in ('sddb', 'chfdb'):
            t = np.round(t * 128.0) / 128.0
        akhir = r.vf_s if r.db == 'sddb' else r.durasi_s
        if r.db == 'sddb' and pd.isna(r.vf_s):
            continue
        for a0 in np.arange(0, akhir - fit.JENDELA + 1, fit.JENDELA):
            m = (t >= a0) & (t < a0 + fit.JENDELA)
            c = fit.ciri(t[m], sym[m])
            if c is None:
                continue
            c.update({'db': r.db, 'rec': r.rec, 'mulai_s': a0,
                      'menit_sebelum_vf': (akhir - (a0 + fit.JENDELA)) / 60 if r.db == 'sddb' else np.nan})
            baris.append(c)
    f = pd.DataFrame(baris)
    f.to_csv(path, index=False)
    return f


def uji_set(f, label):
    r = np.random.default_rng(31)
    nsr, chf = u.ambil_kontrol(f, 'nsrdb', r), u.ambil_kontrol(f, 'chfdb', r)
    v.ULANG = 5
    out = []
    for jenis in ('logistik', 'rf'):
        for h in (0, 60, 360, 720):
            pos = u.ambil_scd(f, h)
            for nama, neg in (('nsrdb', nsr), ('chfdb', chf)):
                sc, subj = v.skor_subjek(pos, neg, 'semua', jenis)
                out.append(dict(set=label, model=jenis, h=h, kontrol=nama, n=pos.rec.nunique(),
                                auc=roc_auc_score(subj, sc)))
        w.ULANG = 5
        d = x.dalam(f, np.random.default_rng(5))
        out.append(dict(set=label, model=jenis, h=-1, kontrol='dalam', n=d.rec.nunique(),
                        auc=roc_auc_score(d.y, w.oof(d, jenis, CIRI))))
    return out


def hz128():
    print('\n== Kuantisasi 128 Hz untuk sddb dan chfdb', flush=True)
    f_asli = x.fitur()
    f128 = u.siapkan(fitur_128())
    res = pd.DataFrame(uji_set(f_asli, 'asli') + uji_set(f128, '128 Hz'))
    res.to_csv(os.path.join(OUT, 'akhir_hz128.csv'), index=False)
    t = res.pivot_table(index=['model', 'kontrol', 'h'], columns='set', values='auc')
    t['selisih'] = t['128 Hz'] - t['asli']
    print(t.round(3).to_string(), flush=True)


# ---------------------------------------------------------------- perpasien
def perpasien():
    f = x.fitur()
    ir = pd.read_csv(os.path.join(OUT, 'irama_jendela.csv'), dtype={'rec': str})
    d = x.dalam(f, np.random.default_rng(5))
    d = d.merge(ir[['db', 'rec', 'mulai_s', 'vt']], on=['db', 'rec', 'mulai_s'], how='left')
    raw = pd.read_csv(os.path.join(OUT, 'fitur.csv'), dtype={'rec': str})[['db', 'rec', 'mulai_s', 'frac_ektopik']]
    d = d.merge(raw.rename(columns={'frac_ektopik': 'ekt_mentah'}), on=['db', 'rec', 'mulai_s'], how='left')
    w.ULANG = 10
    p = w.oof(d, 'rf', CIRI)
    d = d.assign(p=p)
    a0 = roc_auc_score(d.y, d.p)
    rows = []
    for rid, g in d.groupby('rec'):
        dk, jh = g[g.y == 1], g[g.y == 0]
        sisa = d[d.rec != rid]
        rows.append(dict(rec=rid, n_dekat=len(dk), n_jauh=len(jh), selisih=dk.p.mean() - jh.p.mean(),
                         vt_dekat=int(dk.vt.fillna(False).sum()), ekt_dekat=dk.ekt_mentah.mean(),
                         ekt_jauh=jh.ekt_mentah.mean(), auc_tanpa=roc_auc_score(sisa.y, sisa.p)))
    t = pd.DataFrame(rows).sort_values('selisih', ascending=False)
    t.to_csv(os.path.join(OUT, 'akhir_perpasien.csv'), index=False)
    print(f'\n== Rincian per pasien, kontras dalam-pasien RF (AUC semua {a0:.3f})', flush=True)
    print(t.round(3).to_string(index=False), flush=True)
    print(f'  AUC tinggalkan-satu-pasien: {t.auc_tanpa.min():.3f} sampai {t.auc_tanpa.max():.3f}', flush=True)
    vt = t[t.vt_dekat > 0]
    print(f'  pasien dengan jendela VT di 30 menit terakhir: {len(vt)}; selisih rerata {vt.selisih.mean():+.3f} '
          f'lawan {t[t.vt_dekat == 0].selisih.mean():+.3f} tanpa VT', flush=True)
    print(f'  korelasi Spearman selisih skor dengan (ektopi dekat - jauh): '
          f'{t.selisih.corr(t.ekt_dekat - t.ekt_jauh, method="spearman"):.2f}', flush=True)


if __name__ == '__main__':
    pilih = sys.argv[1:] or ['perpasien', 'geser', 'hz128', 'dalam12', 'rf']
    for b in pilih:
        globals()[b]()

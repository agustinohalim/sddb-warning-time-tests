"""Artikel EKG 1 — gambar 1, tiga panel (urutan sesudah paket D, 26 Okt 2026: daya lebih dulu).

A  Simulasi daya (14_paket_b.py b_daya.csv dan 15_paket_d.py d_daya_w.csv, regresi logistik): persen
   simulasi yang menangkap perubahan pra-VF buatan (disesuaikan ukuran, dua sisi, ambang dari 1.000
   simulasi nol b_daya.csv), AUC jendela dalam-pasien lawan uji horizon terhadap nsrdb dan chfdb pada 12
   pasien yang sama; perubahan HRV dalam SD dalam-rekaman lebih dulu, lalu SD gabungan; garis putus 5 %.
B  AUC terhadap horizon sebelum VF (regresi logistik, pembagian tingkat subjek, selang bootstrap
   subjek): SCD lawan nsrdb (semua ciri), lawan chfdb (semua ciri), dan lawan nsrdb dengan
   proporsi denyut ektopik saja. Sumbu mendatar berjarak sama per horizon, bukan skala jam.
C  Kontras dalam-pasien: jendela dekat VF lawan jendela pasien yang sama jauh sebelumnya, untuk
   regresi logistik dan random forest, dengan selang 95 % (hasil_dalam_pasien.csv untuk
   0-30 menit lawan >= 2, 3, 6 jam; revisi_panel.log untuk pita 10 menit lawan >= 2 jam). Pita abu-abu:
   rentang 95 % AUC kontrol pergeseran RF pada pasien yang sama (b_geser_rf.csv dari 14_paket_b.py,
   200 undian).

Palet: biru, jingga, hijau-toska dari palet kategoris baku; setiap garis diberi label langsung
dan penanda berbeda bentuk karena hijau-toska di bawah kontras 3:1.

Keluaran: Gambar/gambar_ekg1_horizon.png (300 dpi) dan .pdf
Pakai:    python 06_gambar.py
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
GAMBAR = os.path.join(HERE, 'Gambar')
PERMUKAAN, TEKS, TEKS2, KISI = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BIRU, JINGGA, TOSKA = '#2a78d6', '#eb6834', '#1baf7a'
GARIS = [
    ('H_nsr_', 'semua', BIRU, 'o', 'vs nsrdb'),
    ('K_chf_', 'semua', JINGGA, 's', 'vs chfdb (heart failure, alive)'),
    ('H_nsr_', 'ektopik_saja', TOSKA, '^', 'vs nsrdb, ectopic fraction only'),
]


def rapikan(ax):
    ax.set_facecolor(PERMUKAAN)
    ax.grid(axis='y', color=KISI, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(KISI)
    ax.tick_params(colors=TEKS2, labelsize=8)


def huruf(ax, h):
    ax.text(-0.02, 1.04, h, transform=ax.transAxes, fontsize=11, fontweight='bold', color=TEKS,
            ha='right', va='bottom')


def panel_horizon(ax):
    h = pd.read_csv(os.path.join(OUT, 'hasil_uji.csv'))
    h = h[h.model == 'logistik']
    for awal, ciri, warna, penanda, label in GARIS:
        d = h[h.uji.str.startswith(awal) & (h.ciri == ciri)].sort_values('horizon_menit')
        x = pd.Series(range(len(d)))
        if ciri == 'semua':
            ax.fill_between(x, d.lo.to_numpy(), d.hi.to_numpy(), color=warna, alpha=0.10, linewidth=0)
        ax.plot(x, d.auc.to_numpy(), color=warna, linewidth=2, marker=penanda, markersize=7,
                markeredgecolor=PERMUKAAN, markeredgewidth=1.5, zorder=3)
        ax.annotate(label, (x.iloc[-1], d.auc.iloc[-1]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=TEKS)
    n = h[h.uji.str.startswith('H_nsr_') & (h.ciri == 'semua')].sort_values('horizon_menit')
    ax.axhline(0.5, color=TEKS2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.1, 0.515, 'chance', fontsize=8, color=TEKS2)
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels([f'{m / 60:g}\nn={k}' for m, k in zip(n.horizon_menit, n.n_pos)])
    ax.set_xlabel('Hours before VF onset (n = sddb patients available)', fontsize=9, color=TEKS)
    ax.set_ylabel('AUC, patient-level', fontsize=9, color=TEKS)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(-0.3, len(n) - 0.7)
    rapikan(ax)
    huruf(ax, 'B')


def panel_dalam(ax):
    dp = pd.read_csv(os.path.join(OUT, 'hasil_dalam_pasien.csv'))
    dp = dp[dp.ciri == 'semua']
    log = open(os.path.join(OUT, 'revisi_panel.log'), encoding='utf-8').read()
    pita = re.findall(r'^\s*(\d+)-(\d+) menit (logistik|rf)\s+AUC ([\d.]+) \[([\d.]+), ([\d.]+)\]', log, re.M)
    baris = []   # (label, model, auc, lo, hi)
    for a, b, m, auc, lo, hi in pita:
        baris.append((f'{a}–{b}\n≥ 2 h', m, float(auc), float(lo), float(hi)))
    for jauh in (120, 180, 360):
        for _, r in dp[dp.jauh == jauh].iterrows():
            baris.append((f'0–30\n≥ {jauh // 60} h', r.model, r.auc, r.lo, r.hi))
    urutan = list(dict.fromkeys(b[0] for b in baris))
    for m, warna, penanda, geser, nama in (('logistik', BIRU, 'o', -0.12, 'Logistic'),
                                          ('rf', JINGGA, 's', 0.12, 'Random forest')):
        d = [b for b in baris if b[1] == m]
        x = [urutan.index(b[0]) + geser for b in d]
        y = [b[2] for b in d]
        ax.errorbar(x, y, yerr=[[b[2] - b[3] for b in d], [b[4] - b[2] for b in d]], fmt=penanda,
                    color=warna, markersize=6, markeredgecolor=PERMUKAAN, elinewidth=1.4, capsize=0,
                    label=nama, zorder=3)
    ax.axhline(0.5, color=TEKS2, linewidth=1, linestyle=(0, (4, 3)))
    gs_ = pd.read_csv(os.path.join(OUT, 'b_geser_rf.csv')).auc
    ax.axhspan(gs_.quantile(0.025), gs_.quantile(0.975), color=TEKS2, alpha=0.10, linewidth=0, zorder=0)
    ax.text(-0.4, 0.34, 'shaded: drift control, 95 % range (random forest)', ha='left',
            va='center', fontsize=7, color=TEKS2)
    ax.set_xticks(range(len(urutan)))
    ax.set_xticklabels(urutan, fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel('AUC, same patients', fontsize=9, color=TEKS)
    ax.set_xlabel('Near windows (min before VF) vs far windows', fontsize=9, color=TEKS)
    ax.legend(frameon=False, fontsize=8, loc='upper left', labelcolor=TEKS)
    rapikan(ax)
    huruf(ax, 'C')


def panel_daya(ax):
    h = pd.concat([pd.read_csv(os.path.join(OUT, 'b_daya.csv')), pd.read_csv(os.path.join(OUT, 'd_daya_w.csv'))])
    n = h[h.jenis == 'nol']
    ambang = {c: n[c].quantile([0.025, 0.975]).to_numpy() for c in ('d_auc', 'h_nsr', 'h_chf')}
    sel = [('searah_w', 1.0, 'HRV 1 SD within'), ('searah_w', 2.0, 'HRV 2 SD within'),
           ('searah_w', 4.0, 'HRV 4 SD within'), ('tegak_w', 2.0, 'HRV 2 SD within,\northogonal'),
           ('searah_k', 1.0, 'HRV 1 SD pooled'), ('searah_k', 2.0, 'HRV 2 SD pooled'),
           ('nadi', 20, 'HR +20/min'), ('ektopi', 4.0, 'ectopy ×4')]
    daya = {c: [] for c in ambang}
    lab = []
    for j, b_, l in sel:
        g = h[(h.jenis == j) & (h.besar == b_)]
        for c, (lo, hi) in ambang.items():
            daya[c].append(((g[c] < lo) | (g[c] > hi)).mean() * 100)
        lab.append(l)
    yy = np.arange(len(sel))[::-1]
    ax.barh(yy + 0.27, daya['d_auc'], height=0.26, color=TOSKA, label='within-patient AUC')
    ax.barh(yy, daya['h_nsr'], height=0.26, color=BIRU, label='horizon vs nsrdb')
    ax.barh(yy - 0.27, daya['h_chf'], height=0.26, color=JINGGA, label='horizon vs chfdb')
    ax.axvline(5, color=TEKS2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(yy)
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Detected (% of simulations)', fontsize=9, color=TEKS)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=TEKS)
    ax.grid(axis='x', color=KISI, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(KISI)
    ax.tick_params(colors=TEKS2, labelsize=8)
    huruf(ax, 'A')


def main():
    os.makedirs(GAMBAR, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 9.6), facecolor=PERMUKAAN)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1, 1], hspace=0.5)
    panel_daya(fig.add_subplot(gs[0]))
    panel_horizon(fig.add_subplot(gs[1]))
    panel_dalam(fig.add_subplot(gs[2]))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.07, top=0.97)
    # panel B: beri ruang label kanan dengan menyempitkan sumbu
    ax = fig.axes[1]
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.66, pos.height])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(GAMBAR, f'gambar_ekg1_horizon.{ext}'), dpi=300, facecolor=PERMUKAAN)
    print('tersimpan', GAMBAR)


if __name__ == '__main__':
    main()

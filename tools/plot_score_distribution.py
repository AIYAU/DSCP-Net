#!/usr/bin/env python3
"""Calibrated Unknown Score Distribution — IEEE TGRS 论文版 1×4 子图

Usage:
    python tools/plot_score_distribution.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPZ_DIR = ROOT / 'outputs' / 'score_distribution'
OUT_DIR = ROOT / 'outputs' / 'figures'

DATASETS = [
    {'key': 'HT',  'name': 'Houston',          'short': '(a) Houston'},
    {'key': 'IP',  'name': 'Indian Pines',      'short': '(b) Indian Pines'},
    {'key': 'LK',  'name': 'WHU-Hi-LongKou',    'short': '(c) WHU-Hi-LongKou'},
    {'key': 'UP',  'name': 'Pavia University',  'short': '(d) Pavia University'},
]

HIST_BINS = 80
KNOWN_COLOR   = '#2166ac'   # muted academic blue
UNKNOWN_COLOR = '#b2182b'   # muted academic red
THRESH_COLOR  = '#4daf4a'   # green
HIST_ALPHA = 0.22
KDE_LW = 1.9

# scipy check
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def load_data(npz_path):
    data = np.load(npz_path)
    return {
        'scores': data['scores'].astype(np.float64),
        'labels': data['labels'].astype(np.int64),
        'threshold': float(data['threshold']),
        'known_classes': data['known_classes'],
        'unknown_classes': data['unknown_classes'],
        'dataset': str(data['dataset']),
    }


def plot_single(ax, data, xlim):
    scores = data['scores']
    labels = data['labels']
    threshold = data['threshold']
    num_known = len(data['known_classes'])
    known_mask = labels < num_known
    known_scores = scores[known_mask]
    unknown_scores = scores[~known_mask]

    lo, hi = xlim

    bins = np.linspace(lo, hi, HIST_BINS)

    # ---- Known ----
    ax.hist(known_scores, bins=bins, density=True, alpha=HIST_ALPHA,
            color=KNOWN_COLOR, edgecolor='none')
    if HAS_SCIPY and len(known_scores) >= 3:
        try:
            kde = gaussian_kde(known_scores)
            xs = np.linspace(lo, hi, 400)
            ax.plot(xs, kde(xs), color=KNOWN_COLOR, linewidth=KDE_LW, label='Known')
        except Exception:
            ax.plot([], [], color=KNOWN_COLOR, linewidth=KDE_LW, label='Known')
    else:
        ax.plot([], [], color=KNOWN_COLOR, linewidth=KDE_LW, label='Known')

    # ---- Unknown ----
    ax.hist(unknown_scores, bins=bins, density=True, alpha=HIST_ALPHA,
            color=UNKNOWN_COLOR, edgecolor='none')
    if HAS_SCIPY and len(unknown_scores) >= 3:
        try:
            kde = gaussian_kde(unknown_scores)
            xs = np.linspace(lo, hi, 400)
            ax.plot(xs, kde(xs), color=UNKNOWN_COLOR, linewidth=KDE_LW, label='Unknown')
        except Exception:
            ax.plot([], [], color=UNKNOWN_COLOR, linewidth=KDE_LW, label='Unknown')
    else:
        ax.plot([], [], color=UNKNOWN_COLOR, linewidth=KDE_LW, label='Unknown')

    # ---- Threshold ----
    ax.axvline(threshold, color=THRESH_COLOR, linestyle='--',
               linewidth=1.4, label='Threshold')

    # threshold text annotation — place near top-right of the line
    ylim = ax.get_ylim()
    y_text = ylim[1] * 0.92
    ax.text(threshold + 0.08, y_text, f'$\\theta={threshold:.2f}$',
            color=THRESH_COLOR, fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.75))

    ax.set_xlim(lo, hi)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- step 1: determine unified x-range ----
    all_scores = []
    datasets_data = []
    for ds in DATASETS:
        npz_path = NPZ_DIR / f'score_dist_{ds["key"]}.npz'
        if not npz_path.exists():
            print(f'ERROR: {npz_path} not found.')
            sys.exit(1)
        data = load_data(npz_path)
        datasets_data.append(data)
        all_scores.append(data['scores'])

    all_scores = np.concatenate(all_scores)
    x_lo = np.quantile(all_scores, 0.005)
    x_hi = np.quantile(all_scores, 0.995)
    x_lo = np.floor(x_lo * 10) / 10    # snap to 0.1
    x_hi = np.ceil(x_hi * 10) / 10
    xlim = (x_lo, x_hi)

    # ---- step 2: sanity check all datasets ----
    expected = {
        'HT': (0.86, 0.09),
        'IP': (0.42, 0.16),
        'LK': (0.94, 0.07),
        'UP': (0.98, 0.05),
    }
    for ds, data in zip(DATASETS, datasets_data):
        scores = data['scores']
        labels = data['labels']
        threshold = data['threshold']
        num_known = len(data['known_classes'])
        known_mask = labels < num_known
        unknown_mask = ~known_mask
        unknown_acc = np.mean(scores[unknown_mask] > threshold)
        known_reject = np.mean(scores[known_mask] > threshold)
        exp_u, exp_k = expected[ds['key']]
        if abs(unknown_acc - exp_u) > 0.25 or abs(known_reject - exp_k) > 0.15:
            print(f'SANITY FAIL: {ds["key"]} UnknownAcc={unknown_acc*100:.1f}% (expected ~{exp_u*100:.0f}%) '
                  f'KRR={known_reject*100:.1f}% (expected ~{exp_k*100:.0f}%)')
            print('  Stopping. Check label mapping / score sign / threshold direction.')
            sys.exit(1)
        print(f'{ds["key"]}: UnknownAcc={unknown_acc*100:.1f}%  KRR={known_reject*100:.1f}%  ✓')

    # ---- step 3: build figure ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    plt.subplots_adjust(wspace=0.28, left=0.05, right=0.98, bottom=0.16, top=0.82)

    for i, (ds, data) in enumerate(zip(DATASETS, datasets_data)):
        ax = axes[i]
        plot_single(ax, data, xlim)

        # Title
        ax.set_title(ds['short'], fontsize=10, fontweight='bold', pad=8)

        # X label on each
        ax.set_xlabel('Calibrated unknown score $S_i$', fontsize=9)

        # Y label only on first
        if i == 0:
            ax.set_ylabel('Density', fontsize=9)
        else:
            ax.set_ylabel('')
            # hide y tick labels for cleaner look on inner plots
            # ax.set_yticklabels([])  -- keep them for context

    # ---- global legend at top centre ----
    # Collect handles from the last subplot (all have same plotted elements)
    handles = []
    # Create proxy artists
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=KNOWN_COLOR, alpha=0.45, label='Known'))
    handles.append(Patch(facecolor=UNKNOWN_COLOR, alpha=0.45, label='Unknown'))
    handles.append(Line2D([0], [0], color=THRESH_COLOR, linestyle='--', linewidth=1.4,
                          label='Threshold'))
    fig.legend(handles=handles, labels=['Known', 'Unknown', 'Threshold'],
               loc='upper center', ncol=3, frameon=False,
               columnspacing=1.2, handlelength=1.5, handletextpad=0.5,
               fontsize=9, bbox_to_anchor=(0.50, 0.97))

    # ---- save ----
    pdf_path = OUT_DIR / 'score_distribution_final.pdf'
    png_path = OUT_DIR / 'score_distribution_final.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.05)
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.05)
    plt.close(fig)

    print(f'\nSaved: {pdf_path}')
    print(f'Saved: {png_path}')
    print(f'  x-range: [{x_lo:.1f}, {x_hi:.1f}]')
    print(f'  Size: 16 × 3.6 in')


if __name__ == '__main__':
    main()

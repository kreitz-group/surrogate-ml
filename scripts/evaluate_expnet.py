"""
Evaluate a trained +/- ExpNet run from scripts/train_expnet.py.

The best model is integrated with dopri8 for every condition of the dataset, and the results
are compared with the observed profiles. Written to <run_dir>/eval/:

    metrics.csv            per split and species: mean relative error of the outlet mole
                           fraction, and mean absolute error over the whole profile
    predictions.csv        observed and predicted mole fractions at every axial point
    parity_outlet.png      predicted vs observed outlet mole fractions, by split
    profiles/<name>.png    one plot per condition: model (lines) vs. data (markers), with the
                           condition's split in the title (--profiles picks which splits)
    history.png            training and validation loss of the best seed
    source_terms.png       learned vs Cantera source terms (only with --cantera, see
                           scripts/expnet_cantera_source_terms.py)

Example:
    python scripts/evaluate_expnet.py results/expnet/generated/paper
    python scripts/evaluate_expnet.py results/expnet/generated/paper \
        --cantera results/expnet/cantera_source_terms_generated.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from expnet import data as D  # noqa: E402
from expnet.reactor import mole_fractions  # noqa: E402
from expnet.train import load_model  # noqa: E402

LABELS = {'CH4': r'CH$_4$', 'CO2': r'CO$_2$', 'CO': 'CO', 'H2': r'H$_2$', 'H2O': r'H$_2$O', 'N2': r'N$_2$'}
PROFILE_SPECIES = ['CH4', 'CO2', 'CO', 'H2O']
SPLIT_COLORS = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}


def style(plt):
    plt.rcParams['figure.figsize'] = (14, 5)
    plt.rcParams['axes.linewidth'] = 2
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['axes.formatter.limits'] = (-3, 6)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--model', default='best_model.pt', help='model file inside run_dir')
    p.add_argument('--data-root', default=None, help='methanation-model repo (default: from config.json)')
    p.add_argument('--cantera', type=Path, default=None, help='CSV from expnet_cantera_source_terms.py')
    p.add_argument('--profiles', nargs='+', choices=['all', 'train', 'val', 'test'], default=['all'],
                   help='splits to draw a profile plot for, one PNG per condition (default: all)')
    p.add_argument('--show', action='store_true', help='plt.show() at the end')
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / 'eval'
    out_dir.mkdir(exist_ok=True)

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)

    cfg = json.loads((run_dir / 'config.json').read_text())
    split = json.loads((run_dir / 'split.json').read_text())
    data_root = args.data_root or cfg['data_root']

    reactor = D.Reactor()
    kwargs = {'obs_stride': 1} if cfg['dataset'] == 'generated' else {}
    data = D.load(cfg['dataset'], data_root, reactor, **kwargs)
    pfr, ckpt = load_model(run_dir / args.model)
    print(f'{run_dir.name}: seed {ckpt.get("seed")}, {ckpt["n_parameters"]} parameters, '
          f'key species {ckpt["key_species"]}')

    with torch.no_grad():
        n_pred = pfr.solve(data, method='dopri8', rtol=cfg['rtol'], atol=cfg['atol'])
        y_pred = mole_fractions(n_pred).numpy()
    y_obs = data.y_obs.numpy()
    z_mm = data.z_obs.numpy() * 1e3

    split_of = {name: s for s, names in split.items() for name in names}
    splits = np.array([split_of.get(name, 'unused') for name in data.names])

    # ---- metrics
    species = cfg['loss_species']
    rows = []
    for s_name in ['train', 'val', 'test']:
        m = splits == s_name
        if not m.any():
            continue
        for sp in species:
            i = D.SPECIES.index(sp)
            yo, yp = y_obs[m, :, i], y_pred[m, :, i]
            rows.append({
                'split': s_name, 'species': sp,
                'outlet_mean_rel_err': np.mean(np.abs(yp[:, -1] - yo[:, -1]) / np.abs(yo[:, -1])),
                'outlet_median_rel_err': np.median(np.abs(yp[:, -1] - yo[:, -1]) / np.abs(yo[:, -1])),
                'profile_mean_abs_err': np.mean(np.abs(yp - yo)),
                'profile_rel_err_vs_mean': np.mean(np.abs(yp - yo)) / np.mean(np.abs(yo)),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'metrics.csv', index=False)
    with pd.option_context('display.float_format', '{:.3e}'.format):
        print(metrics.to_string(index=False))

    # ---- predictions table
    B, O, S = y_obs.shape
    pred = pd.DataFrame({
        'condition': np.repeat(data.names, O),
        'split': np.repeat(splits, O),
        'z_mm': np.tile(z_mm, B),
    })
    for sp in D.SPECIES:
        i = D.SPECIES.index(sp)
        pred[f'y_{sp}_obs'] = y_obs[:, :, i].ravel()
        pred[f'y_{sp}_pred'] = y_pred[:, :, i].ravel()
    pred.to_csv(out_dir / 'predictions.csv', index=False)

    # ---- outlet parity
    fig, axes = plt.subplots(1, len(species), figsize=(4.2 * len(species), 4.4))
    for ax, sp in zip(np.atleast_1d(axes), species):
        i = D.SPECIES.index(sp)
        for s_name, color in SPLIT_COLORS.items():
            m = splits == s_name
            ax.scatter(y_obs[m, -1, i], y_pred[m, -1, i], s=18, color=color, label=s_name, alpha=0.8)
        lo = min(y_obs[:, -1, i].min(), y_pred[:, -1, i].min())
        hi = max(y_obs[:, -1, i].max(), y_pred[:, -1, i].max())
        ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
        ax.plot([lo, hi], [1.2 * lo, 1.2 * hi], 'k--', lw=0.8)
        ax.plot([lo, hi], [0.8 * lo, 0.8 * hi], 'k--', lw=0.8)
        ax.set_title(LABELS[sp], fontsize=16)
        ax.set_xlabel(r'observed $y_\mathrm{out}$')
    np.atleast_1d(axes)[0].set_ylabel(r'predicted $y_\mathrm{out}$')
    np.atleast_1d(axes)[0].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'parity_outlet.png', dpi=150)

    # ---- one profile plot per condition
    obs_label = 'experiment' if cfg['dataset'] == 'experimental' else 'Cantera (mech. 4758)'
    plot_splits = ['train', 'val', 'test'] if 'all' in args.profiles else args.profiles
    prof_dir = out_dir / 'profiles'
    prof_dir.mkdir(exist_ok=True)
    for old in prof_dir.glob('*.png'):
        old.unlink()
    stride = max(1, len(z_mm) // 20)   # generated profiles have 201 points; show about 20 markers
    n_plotted = 0
    for b, name in enumerate(data.names):
        if splits[b] not in plot_splits:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        species_handles = []
        for k, sp in enumerate(PROFILE_SPECIES):
            i = D.SPECIES.index(sp)
            color = f'C{k}'
            ax.plot(z_mm, y_pred[b, :, i], '-', color=color, lw=2)
            ax.plot(z_mm[::stride], y_obs[b, ::stride, i], 'o', color=color, mfc='none', mew=1.5)
            species_handles.append(Line2D([], [], color=color, lw=6, label=LABELS[sp]))
        source_handles = [
            Line2D([], [], color='k', ls='-', lw=2, label='+/- ExpNet model'),
            Line2D([], [], color='k', ls='none', marker='o', mfc='none', mew=1.5, label=obs_label),
        ]
        T_lo, T_hi = float(data.T[b].min()) - 273.15, float(data.T[b].max()) - 273.15
        T_txt = f'{T_lo:.0f} °C' if T_hi - T_lo < 1 else f'{T_lo:.0f}-{T_hi:.0f} °C'
        ax.set_title(f'{name} ({splits[b]}):  {T_txt},  {float(data.P_atm[b]) * 1.01325:.1f} bar', fontsize=13)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('mole fraction')
        leg1 = ax.legend(handles=species_handles, fontsize=11, loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.add_artist(leg1)
        ax.legend(handles=source_handles, fontsize=11, loc='lower left', bbox_to_anchor=(1.01, 0))
        fig.tight_layout()
        fig.savefig(prof_dir / f'{name}.png', dpi=150)
        plt.close(fig)
        n_plotted += 1
    print(f'Wrote {n_plotted} profile plots to {prof_dir}')

    # ---- history
    seed = ckpt.get('seed')
    hist_file = run_dir / f'seed_{seed}' / 'history.csv'
    if hist_file.exists():
        h = pd.read_csv(hist_file)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(h))
        ax.semilogy(x, h['train'], lw=1, label='train')
        ax.semilogy(x, h['val'], lw=1, label='val')
        for stage, g in h.groupby('stage', sort=False):
            ax.axvline(g.index[0], color='grey', lw=0.5)
            ax.text(g.index[0], ax.get_ylim()[1], stage, rotation=90, va='top', fontsize=8)
        ax.set_xlabel('epoch (all stages)')
        ax.set_ylabel('weighted MSE')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'history.png', dpi=150)

    # ---- learned vs Cantera source terms
    if args.cantera is not None:
        compare_source_terms(pfr, args.cantera, split_of, out_dir, plt)

    print(f'Wrote evaluation to {out_dir}')
    if args.show:
        plt.show()


def compare_source_terms(pfr, csv_file, split_of, out_dir, plt):
    """Parity of the learned source terms against Cantera steady-state net production rates."""

    ct_df = pd.read_csv(csv_file)
    T = torch.as_tensor(ct_df['T_K'].to_numpy(), dtype=torch.float64)
    x = ct_df[[f'x_{s}' for s in D.REACTIVE]].to_numpy()
    p = torch.as_tensor(x * ct_df[['P_atm']].to_numpy(), dtype=torch.float64)

    with torch.no_grad():
        s_pred = pfr.kinetics(T, p).numpy()

    split = ct_df['condition'].map(split_of).fillna('unused').to_numpy()

    rows = []
    fig, axes = plt.subplots(1, len(D.REACTIVE), figsize=(4.2 * len(D.REACTIVE), 4.4))
    for ax, sp in zip(axes, D.REACTIVE):
        k = D.REACTIVE.index(sp)
        s_true = ct_df[f's_{sp}'].to_numpy()
        for s_name, color in SPLIT_COLORS.items():
            m = split == s_name
            ax.scatter(s_true[m], s_pred[m, k], s=8, color=color, alpha=0.6, label=s_name)
            if m.any():
                rel = np.abs(s_pred[m, k] - s_true[m]) / np.abs(s_true[m])
                rows.append({'split': s_name, 'species': sp, 'mean_rel_err': rel.mean(),
                             'median_rel_err': np.median(rel)})
        lo, hi = s_true.min(), s_true.max()
        ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
        ax.set_title(LABELS[sp], fontsize=16)
        ax.set_xlabel(r'Cantera $\dot{s}$ (mol m$^{-2}$ s$^{-1}$)')
    axes[0].set_ylabel(r'+/- ExpNet $\dot{s}$ (mol m$^{-2}$ s$^{-1}$)')
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'source_terms.png', dpi=150)

    st = pd.DataFrame(rows)
    st.to_csv(out_dir / 'source_term_metrics.csv', index=False)
    with pd.option_context('display.float_format', '{:.3e}'.format):
        print('Learned vs Cantera source terms:')
        print(st.to_string(index=False))


if __name__ == '__main__':
    main()

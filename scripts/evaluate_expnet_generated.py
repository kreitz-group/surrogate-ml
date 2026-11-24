"""
Test a trained +/- ExpNet on the 200 generated Ni/SiO2 profiles of the methanation-model repo
(results/NiSiO2_generated_conditions_mechanism_4758.h5, simulated with mechanism 4758) and
plot the predicted vs. simulated profile of every condition.

Any trained model can be tested:
  - a model trained on the generated data: every condition is labelled with the split it was
    in during training (train / val / test, read from split.json next to the model). Judge the
    model on the test conditions.
  - any other model (e.g. trained on the experimental profiles): all conditions are labelled
    "unseen". Conditions outside the model's training range of T and p are extrapolations.

The model is integrated with dopri8 from each condition's inlet, at the condition's
temperature and pressure. Written to <out>/ (default: <run_dir>/eval_generated/):

    profiles/condition_<k>.png   one plot per condition: CH4, CO2, CO, H2O (left) and H2
                                 (right), model as lines and Cantera simulation as markers
    condition_errors.csv         per condition: T, p, inlet composition, split, outlet relative
                                 error per species and mean absolute profile error
    metrics.csv                  per split and species: mean / median outlet relative error,
                                 mean absolute profile error
    predictions.csv              simulated and predicted mole fractions at every axial point
    parity_outlet.png            predicted vs simulated outlet mole fractions
    error_map.png                outlet relative error of each condition over T and p, per species

Relative outlet errors blow up where a mole fraction is close to zero (CO2 and CO near full
conversion, CH4 at low temperature), so read them per species next to the profile plots.

Examples:
    python scripts/evaluate_expnet_generated.py results/expnet/generated/laptop_1seed
    python scripts/evaluate_expnet_generated.py results/expnet/generated/paper/seed_3/model.pt
    python scripts/evaluate_expnet_generated.py results/expnet/experimental/laptop_1seed
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from expnet import data as D  # noqa: E402
from expnet.reactor import mole_fractions  # noqa: E402
from expnet.train import load_model  # noqa: E402

LABELS = {'CH4': r'CH$_4$', 'CO2': r'CO$_2$', 'CO': 'CO', 'H2': r'H$_2$', 'H2O': r'H$_2$O', 'N2': r'N$_2$'}
MINOR_SPECIES = ['CH4', 'CO2', 'CO', 'H2O']   # left panel; H2 is ~10x larger and gets its own panel
ERROR_SPECIES = ['CH4', 'CO2', 'CO', 'H2', 'H2O']
SPLIT_COLORS = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green', 'unseen': 'tab:purple'}
DEFAULT_RUN = REPO_ROOT / 'results' / 'expnet' / 'generated' / 'laptop_1seed'


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
    p.add_argument('model', type=Path, nargs='?', default=DEFAULT_RUN,
                   help='run folder (uses its best_model.pt) or a model .pt file '
                        f'(default: {DEFAULT_RUN.relative_to(REPO_ROOT)})')
    p.add_argument('--data-root', type=Path, default=D.DEFAULT_DATA_ROOT, help='path to the methanation-model repo')
    p.add_argument('--mechanism', type=int, default=4758)
    p.add_argument('--out', type=Path, default=None, help='output folder (default: <run_dir>/eval_generated)')
    p.add_argument('--no-profiles', action='store_true', help='skip the 200 per-condition plots')
    p.add_argument('--show', action='store_true', help='plt.show() the summary plots at the end')
    return p.parse_args()


def find_run_files(model_arg):
    """Model file, and the run folder holding config.json / split.json (if any)."""
    model_arg = model_arg.resolve()
    model_file = model_arg / 'best_model.pt' if model_arg.is_dir() else model_arg
    if not model_file.exists():
        raise SystemExit(f'No model file at {model_file}')
    # best_model.pt sits in the run folder, seed_<k>/model.pt one level below it
    for folder in [model_file.parent, model_file.parent.parent]:
        if (folder / 'config.json').exists():
            return model_file, folder
    return model_file, None


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)

    model_file, run_dir = find_run_files(args.model)
    cfg = json.loads((run_dir / 'config.json').read_text()) if run_dir else {}
    out_dir = (args.out or (run_dir or model_file.parent) / 'eval_generated').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # split labels only mean something if the model was trained on these same profiles
    split_of = {}
    trained_on = cfg.get('dataset', 'unknown')
    if trained_on == 'generated' and (run_dir / 'split.json').exists():
        split = json.loads((run_dir / 'split.json').read_text())
        split_of = {name: s for s, names in split.items() for name in names}

    data = D.load_generated(args.data_root, D.Reactor(), mechanism=args.mechanism, obs_stride=1)
    pfr, ckpt = load_model(model_file)
    print(f'Model {model_file} (trained on: {trained_on}, seed {ckpt.get("seed")}, '
          f'{ckpt["n_parameters"]} parameters)')
    print(f'Testing on {len(data)} generated conditions x {data.z_obs.numel()} axial points '
          f'(mechanism {args.mechanism})')

    with torch.no_grad():
        n_pred = pfr.solve(data, method='dopri8', rtol=cfg.get('rtol', 1e-6), atol=cfg.get('atol', 1e-10))
        y_pred = mole_fractions(n_pred).numpy()
    y_obs = data.y_obs.numpy()
    z_mm = data.z_obs.numpy() * 1e3
    splits = np.array([split_of.get(name, 'unseen') for name in data.names])
    T_C = data.T.mean(dim=1).numpy() - 273.15
    p_bar = data.P_atm.numpy() * 1.01325
    x_in = (data.n_in / data.n_in.sum(dim=1, keepdim=True)).numpy()

    if not np.isfinite(y_pred).all():
        bad = [data.names[b] for b in range(len(data)) if not np.isfinite(y_pred[b]).all()]
        print(f'Warning: non-finite predictions for {len(bad)} conditions: {bad[:10]}')

    # ---- errors per condition
    idx = [D.SPECIES.index(sp) for sp in ERROR_SPECIES]
    rel_out = np.abs(y_pred[:, -1, idx] - y_obs[:, -1, idx]) / np.abs(y_obs[:, -1, idx])
    mae_prof = np.mean(np.abs(y_pred[:, :, idx] - y_obs[:, :, idx]), axis=1)
    cond = pd.DataFrame({'condition': data.names, 'split': splits, 'T_C': T_C, 'p_bar': p_bar})
    for sp in ['H2', 'CO2', 'N2']:
        cond[f'y_{sp}_in'] = x_in[:, D.SPECIES.index(sp)]
    for k, sp in enumerate(ERROR_SPECIES):
        cond[f'outlet_rel_err_{sp}'] = rel_out[:, k]
    for k, sp in enumerate(ERROR_SPECIES):
        cond[f'profile_mae_{sp}'] = mae_prof[:, k]
    cond['outlet_rel_err_mean'] = rel_out.mean(axis=1)
    cond.to_csv(out_dir / 'condition_errors.csv', index=False)

    # ---- metrics per split
    rows = []
    for s_name in ['train', 'val', 'test', 'unseen']:
        m = splits == s_name
        if not m.any():
            continue
        for k, sp in enumerate(ERROR_SPECIES):
            rows.append({'split': s_name, 'n_conditions': int(m.sum()), 'species': sp,
                         'outlet_mean_rel_err': rel_out[m, k].mean(),
                         'outlet_median_rel_err': np.median(rel_out[m, k]),
                         'profile_mean_abs_err': mae_prof[m, k].mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'metrics.csv', index=False)
    with pd.option_context('display.float_format', '{:.3e}'.format):
        print(metrics.to_string(index=False))
    print('Largest outlet relative error per species (condition, split, error):')
    for sp in ERROR_SPECIES:
        w = cond.sort_values(f'outlet_rel_err_{sp}', ascending=False).head(3)
        print(f'  {sp:4s} ' + ',  '.join(f'{r.condition} ({r.split}) {100 * r[f"outlet_rel_err_{sp}"]:.1f}%'
                                          for _, r in w.iterrows()))

    # ---- predictions table
    B, O, _ = y_obs.shape
    pred = pd.DataFrame({'condition': np.repeat(data.names, O), 'split': np.repeat(splits, O),
                         'z_mm': np.tile(z_mm, B)})
    for sp in D.SPECIES:
        i = D.SPECIES.index(sp)
        pred[f'y_{sp}_sim'] = y_obs[:, :, i].ravel()
        pred[f'y_{sp}_pred'] = y_pred[:, :, i].ravel()
    pred.to_csv(out_dir / 'predictions.csv', index=False)

    present = [s for s in SPLIT_COLORS if (splits == s).any()]

    # ---- outlet parity
    fig, axes = plt.subplots(1, len(ERROR_SPECIES), figsize=(4.2 * len(ERROR_SPECIES), 4.4))
    for ax, sp in zip(axes, ERROR_SPECIES):
        i = D.SPECIES.index(sp)
        for s_name in present:
            m = splits == s_name
            ax.scatter(y_obs[m, -1, i], y_pred[m, -1, i], s=18, color=SPLIT_COLORS[s_name], label=s_name, alpha=0.8)
        lo = np.nanmin([y_obs[:, -1, i].min(), np.nanmin(y_pred[:, -1, i])])
        hi = np.nanmax([y_obs[:, -1, i].max(), np.nanmax(y_pred[:, -1, i])])
        ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
        ax.plot([lo, hi], [1.2 * lo, 1.2 * hi], 'k--', lw=0.8)
        ax.plot([lo, hi], [0.8 * lo, 0.8 * hi], 'k--', lw=0.8)
        ax.set_title(LABELS[sp], fontsize=16)
        ax.set_xlabel(r'Cantera $y_\mathrm{out}$')
    axes[0].set_ylabel(r'+/- ExpNet $y_\mathrm{out}$')
    axes[0].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'parity_outlet.png', dpi=150)

    # ---- where in (T, p) the model is worst
    markers = {'train': 'o', 'val': 's', 'test': '^', 'unseen': 'o'}
    fig, axes = plt.subplots(1, len(ERROR_SPECIES), figsize=(5.2 * len(ERROR_SPECIES), 4.8), sharey=True)
    for ax, sp in zip(axes, ERROR_SPECIES):
        err = 100 * cond[f'outlet_rel_err_{sp}'].to_numpy()
        vmax = np.nanpercentile(err, 95)   # colour scale capped so a few outliers don't wash it out
        for s_name in present:
            m = splits == s_name
            sc = ax.scatter(T_C[m], p_bar[m], c=err[m], cmap='viridis', vmin=0, vmax=vmax,
                            marker=markers[s_name], s=40, edgecolors='k', linewidths=0.5)
        fig.colorbar(sc, ax=ax, extend='max').set_label('outlet rel. error (%)', fontsize=12)
        ax.set_title(LABELS[sp], fontsize=16)
        ax.set_xlabel('T (°C)')
    axes[0].set_ylabel('p (bar)')
    if len(present) > 1:
        axes[0].legend(handles=[Line2D([], [], color='grey', ls='none', marker=markers[s], mec='k', label=s)
                                for s in present], fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_map.png', dpi=150)

    # ---- one profile plot per condition
    if not args.no_profiles:
        prof_dir = out_dir / 'profiles'
        prof_dir.mkdir(exist_ok=True)
        for old in prof_dir.glob('*.png'):
            old.unlink()
        stride = max(1, len(z_mm) // 20)   # 201 simulated points; show about 20 markers
        source_handles = [
            Line2D([], [], color='k', ls='-', lw=2, label='+/- ExpNet model'),
            Line2D([], [], color='k', ls='none', marker='o', mfc='none', mew=1.5, label='Cantera simulation'),
        ]
        for b, name in enumerate(data.names):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
            species_handles = []
            for k, sp in enumerate(MINOR_SPECIES + ['H2']):
                i = D.SPECIES.index(sp)
                ax = ax2 if sp == 'H2' else ax1
                color = f'C{k}'
                ax.plot(z_mm, y_pred[b, :, i], '-', color=color, lw=2)
                ax.plot(z_mm[::stride], y_obs[b, ::stride, i], 'o', color=color, mfc='none', mew=1.5)
                species_handles.append(Line2D([], [], color=color, lw=6, label=LABELS[sp]))
            for ax in (ax1, ax2):
                ax.set_xlabel('z (mm)')
            ax1.set_ylabel('mole fraction')
            ax1.legend(handles=species_handles[:-1], fontsize=11, loc='best')
            ax2.legend(handles=species_handles[-1:] + source_handles, fontsize=11, loc='best')
            fig.suptitle(
                f'{name} ({splits[b]}):  {T_C[b]:.0f} °C,  {p_bar[b]:.2f} bar,  inlet H$_2$/CO$_2$/N$_2$ = '
                f'{x_in[b, D.SPECIES.index("H2")]:.3f}/{x_in[b, D.SPECIES.index("CO2")]:.3f}/'
                f'{x_in[b, D.SPECIES.index("N2")]:.3f}\noutlet relative error:  ' + ',  '.join(
                    f'{LABELS[sp]} {100 * rel_out[b, k]:.1f}%' for k, sp in enumerate(ERROR_SPECIES)),
                fontsize=12)
            fig.tight_layout()
            number = int(name.split('_')[1])
            fig.savefig(prof_dir / f'condition_{number:03d}.png', dpi=120)
            plt.close(fig)
        print(f'Wrote {len(data)} profile plots to {prof_dir}')

    print(f'Wrote evaluation to {out_dir}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

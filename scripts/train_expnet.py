"""
Train +/- ExpNet kinetics on Ni/SiO2 profile-reactor data as a neural ODE (Döppel et al. 2026).

Several randomly initialized models (seeds) are trained and the one with the lowest
validation loss after the final dopri8 stage is kept as best_model.pt. Seeds run in
parallel processes with --workers.

Outputs, in results/expnet/<dataset>/<tag>/:
    config.json        settings of the run
    split.json         condition names in the train / val / test sets
    seed_<k>/model.pt  trained model of each seed, seed_<k>/history.csv its loss curves
    summary.csv        final train / val loss of every seed
    best_model.pt      copy of the best seed's model

Defaults follow the paper (1 hidden layer of 50 tanh nodes, RMSprop lr 1e-3, Euler with
1 -> 128 control volumes for 1000 epochs each, then 100 epochs of dopri8, 10 seeds).

Examples:
    # quick check on the laptop (a few minutes)
    python scripts/train_expnet.py --quick

    # full run on the server, 10 seeds in parallel
    python scripts/train_expnet.py --dataset generated --seeds 10 --workers 10 --tag paper

    # the measured Küchen profiles
    python scripts/train_expnet.py --dataset experimental --seeds 10 --workers 10 --tag paper

    # mechanism 4758 simulated at the 24 Küchen conditions (same split as 'experimental')
    python scripts/train_expnet.py --dataset experimental_sim --seeds 10 --workers 10 --tag paper
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from expnet import data as D  # noqa: E402
from expnet.reactor import mole_fractions, species_weights, weighted_mse  # noqa: E402
from expnet.train import Config, train_one, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=D.DATASETS, default='generated',
                   help="'generated' (200 simulated conditions), 'experimental' (24 measured profiles) or "
                        "'experimental_sim' (mechanism 4758 simulated at the 24 experimental conditions)")
    p.add_argument('--data-root', default=str(D.DEFAULT_DATA_ROOT), help='path to the methanation-model repo')
    p.add_argument('--tag', default=None, help='run name (default: timestamp)')
    p.add_argument('--seeds', type=int, default=10, help='number of random initializations')
    p.add_argument('--first-seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=1, help='seeds trained in parallel')
    p.add_argument('--threads', type=int, default=1, help='torch threads per worker')
    p.add_argument('--hidden', type=int, nargs='+', default=[50], help='hidden layer widths')
    p.add_argument('--key-species', nargs='+', default=['CO', 'CH4'])
    p.add_argument('--loss-species', nargs='+', default=['CH4', 'CO2', 'CO', 'H2', 'H2O'])
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs-euler', type=int, default=1000, help='epochs per Euler stage')
    p.add_argument('--max-cv', type=int, default=128, help='largest number of Euler control volumes (power of 2)')
    p.add_argument('--epochs-dopri', type=int, default=100)
    p.add_argument('--out-bias', type=float, default=None, help='initial output bias (default: from data)')
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--obs-stride', type=int, default=1, help='use every n-th axial point (simulated datasets)')
    p.add_argument('--device', default='cpu', help="'cpu' or 'cuda'")
    p.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--quick', action='store_true',
                   help='smoke test: 1 seed, 50 epochs per Euler stage up to 16 CVs, 5 dopri8 epochs')
    return p.parse_args()


def run_seed(cfg, seed, run_dir, threads):
    torch.set_num_threads(threads)
    reactor = D.Reactor()
    data = D.load(cfg.dataset, cfg.data_root, reactor, obs_stride=cfg.obs_stride)
    train_idx, val_idx, _ = D.split_indices(len(data), cfg.split, cfg.split_seed)
    train, val = data.subset(train_idx), data.subset(val_idx)

    seed_dir = run_dir / f'seed_{seed}'
    pfr, history = train_one(cfg, train, val, reactor, seed, out_dir=seed_dir)

    # final losses with dopri8, whatever the last training stage was
    dtype = getattr(torch, cfg.dtype)
    train, val = train.to(cfg.device, dtype), val.to(cfg.device, dtype)
    w = species_weights(train.y_obs, cfg.loss_species)
    with torch.no_grad():
        losses = {}
        for name, d in [('train', train), ('val', val)]:
            n = pfr.solve(d, method='dopri8', rtol=cfg.rtol, atol=cfg.atol)
            losses[name] = float(weighted_mse(mole_fractions(n), d.y_obs, w, cfg.loss_species))

    return {'seed': seed, 'train_loss': losses['train'], 'val_loss': losses['val'],
            'train_time_s': float(history['time_s'].iloc[-1]) if len(history) else 0.0}


def main():
    args = parse_args()

    if args.quick:
        args.seeds, args.epochs_euler, args.max_cv, args.epochs_dopri, args.log_every = 1, 50, 16, 5, 25
        args.tag = args.tag or 'quick'

    cfg = Config(
        dataset=args.dataset, data_root=args.data_root, key_species=args.key_species,
        loss_species=args.loss_species, hidden=args.hidden, lr=args.lr, epochs_euler=args.epochs_euler,
        max_cv=args.max_cv, epochs_dopri=args.epochs_dopri, out_bias=args.out_bias,
        split_seed=args.split_seed, obs_stride=args.obs_stride, dtype=args.dtype, device=args.device,
        log_every=args.log_every,
    )

    if args.max_cv & (args.max_cv - 1):
        raise SystemExit('--max-cv must be a power of 2')

    tag = args.tag or time.strftime('%Y%m%d-%H%M%S')
    run_dir = REPO_ROOT / 'results' / 'expnet' / args.dataset / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # data summary and split, written once by the main process
    data = D.load(cfg.dataset, cfg.data_root, D.Reactor(), obs_stride=cfg.obs_stride)
    train_idx, val_idx, test_idx = D.split_indices(len(data), cfg.split, cfg.split_seed)
    write_json(asdict(cfg), run_dir / 'config.json')
    write_json({k: [data.names[i] for i in idx] for k, idx in
                [('train', train_idx), ('val', val_idx), ('test', test_idx)]}, run_dir / 'split.json')

    print(f'{args.dataset}: {len(data)} conditions x {data.z_obs.numel()} axial points '
          f'(train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)})')
    print(f'Writing to {run_dir}')

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as ex:
            rows = list(ex.map(run_seed, [cfg] * len(seeds), seeds, [run_dir] * len(seeds), [args.threads] * len(seeds)))
    else:
        rows = [run_seed(cfg, s, run_dir, args.threads) for s in seeds]

    summary = pd.DataFrame(rows).sort_values('val_loss')
    summary.to_csv(run_dir / 'summary.csv', index=False)
    print(summary.to_string(index=False))

    best = int(summary['seed'].iloc[0])
    shutil.copy(run_dir / f'seed_{best}' / 'model.pt', run_dir / 'best_model.pt')
    print(f'Best seed {best} -> {run_dir / "best_model.pt"}')


if __name__ == '__main__':
    main()

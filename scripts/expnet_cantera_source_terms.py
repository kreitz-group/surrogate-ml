"""
Ground-truth source terms of mechanism 4758 at reactor states, to check what the +/- ExpNet
learned from the profiles (scripts/evaluate_expnet.py --cantera <this csv>).

The ExpNet never sees source terms during training, only mole-fraction profiles, so this is
an independent check of whether the learned kinetics match the mechanism that generated the
data.

    generated     For every n-th axial point of each generated profile the stored gas state
                  and surface coverages are loaded into the mechanism, and the gas-species net
                  production rates are read off. No steady-state solve is needed because the
                  HDF5 file already holds the steady-state coverages (pass --resolve to solve
                  them again instead).
    experimental_sim  The same, for mechanism 4758 simulated at the 24 experimental conditions
                  (results/NiSiO2_24_conditions.h5), which also stores the coverages.
    experimental  At every measured point of the Küchen profiles the surface is solved to
                  steady state at the measured T, p and composition (as in calc_rates_coverages.py).
                  This compares the ExpNet trained on measurements with mechanism 4758.

Units: mol / m^2 Ni / s (Cantera's kmol/m^2/s x 1000), the same as the ExpNet output.

Output: results/expnet/cantera_source_terms_<dataset>.csv with columns
    condition, z_mm, T_K, P_atm, x_<species>, s_<species>

Run from anywhere:
    python scripts/expnet_cantera_source_terms.py --dataset generated --stride 10
"""

import argparse
import sys
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from expnet import data as D  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=D.DATASETS, default='generated')
    p.add_argument('--data-root', type=Path, default=D.DEFAULT_DATA_ROOT)
    p.add_argument('--mechanism', type=int, default=4758)
    p.add_argument('--stride', type=int, default=10, help='every n-th axial point (simulated datasets)')
    p.add_argument('--resolve', action='store_true', help='solve coverages to steady state (simulated datasets)')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def gas_rates(surf, names):
    """Net production rates of the given gas species in mol/m^2/s."""
    rates = surf.net_production_rates
    return [1e3 * rates[surf.kinetics_species_index(D.MECH_NAMES[s])] for s in names]


def main():
    args = parse_args()
    out = args.out or REPO_ROOT / 'results' / 'expnet' / f'cantera_source_terms_{args.dataset}.csv'

    yaml_file = args.data_root / 'data' / 'mechanism_files' / f'File_{args.mechanism}' / 'chem_covdep_formatted.yaml'

    # one Interface for the whole run; coverages are reset or overwritten for every state
    surf = ct.Interface(yaml_file, name='surface1')
    gas = surf.adjacent['gas']
    initial_coverages = surf.coverages.copy()

    rows, n_failed = [], 0

    if args.dataset in ('generated', 'experimental_sim'):
        import h5py
        reactor = D.Reactor()
        dz_mm = reactor.length / (reactor.n_cstr - 1) * 1e3
        if args.dataset == 'generated':
            h5_file = args.data_root / 'results' / f'NiSiO2_generated_conditions_mechanism_{args.mechanism}.h5'
        else:
            h5_file = args.data_root / 'results' / 'NiSiO2_24_conditions.h5'

        with h5py.File(h5_file, 'r') as f:
            for group in tqdm(sorted(f, key=lambda k: int(k.split('_')[1]))):
                key = f'{group}/mechanism_{args.mechanism}'
                if key not in f:
                    continue
                ds = f[key]
                # condition names as in expnet.data: condition_<k> or exp_<n>
                cond = group if args.dataset == 'generated' else group.removesuffix('_on_NiSiO2')
                prof = pd.DataFrame(ds[:], columns=list(ds.attrs['columns']))
                idx = np.unique(np.r_[np.arange(0, len(prof), args.stride), len(prof) - 1])

                for n in idx:
                    row = prof.iloc[n]
                    T = row['T (C)'] + 273.15
                    P = row['P (atm)'] * ct.one_atm
                    X = {s: max(row[s], 0.0) for s in gas.species_names}

                    surf.TP = T, P
                    gas.TPX = T, P, X
                    if args.resolve:
                        surf.coverages = initial_coverages
                        try:
                            surf.advance_coverages_to_steady_state()
                        except ct.CanteraError:
                            n_failed += 1
                            continue
                    else:
                        surf.coverages = [max(row[s], 0.0) for s in surf.species_names]

                    x = [row[D.MECH_NAMES[s]] for s in D.REACTIVE]
                    rows.append([cond, (n + 1) * dz_mm, T, P / ct.one_atm] + x + gas_rates(surf, D.REACTIVE))

    else:
        data_dir = args.data_root / 'data' / 'Kuchen_Experimental_data'
        for n_exp in tqdm(range(24)):
            df = pd.read_csv(data_dir / f'exp_{n_exp}_data.csv')
            for _, row in df.iterrows():
                T = row['Temp']
                P = row['p'] * 1e5
                y = row[D.SPECIES].to_numpy(dtype=float)
                y = y / y.sum()
                X = {D.MECH_NAMES[s]: max(v, 0.0) for s, v in zip(D.SPECIES, y)}

                surf.TP = T, P
                surf.coverages = initial_coverages
                gas.TPX = T, P, X
                try:
                    surf.advance_coverages_to_steady_state()
                except ct.CanteraError:
                    n_failed += 1
                    continue

                x = [y[D.SPECIES.index(s)] for s in D.REACTIVE]
                rows.append([f'exp_{n_exp}', row['z'], T, P / ct.one_atm] + x + gas_rates(surf, D.REACTIVE))

    cols = (['condition', 'z_mm', 'T_K', 'P_atm'] + [f'x_{s}' for s in D.REACTIVE]
            + [f's_{s}' for s in D.REACTIVE])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    print(f'Wrote {len(rows)} states to {out} ({n_failed} failed)')


if __name__ == '__main__':
    main()

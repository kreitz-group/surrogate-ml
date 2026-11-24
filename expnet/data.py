"""
Reactor profile data for the Ni/SiO2 profile reactor, read from the methanation-model repo.

Three datasets are supported:

    generated     results/NiSiO2_generated_conditions_mechanism_4758.h5
                  200 isothermal conditions simulated with the best-fit mechanism 4758
                  (inlets in data/NiSiO2_generated_conditions.csv), 201 axial points each.
    experimental  data/Kuchen_Experimental_data/exp_{0..23}_data.csv
                  24 measured profiles with a measured axial temperature profile.
    experimental_sim  results/NiSiO2_24_conditions.h5, group exp_<n>_on_NiSiO2/mechanism_4758
                  mechanism 4758 simulated at the 24 experimental conditions (measured inlet
                  and measured T(z) imposed), 201 axial points each. Same condition names
                  (exp_<n>) as 'experimental', so the same --split-seed gives the same split.

All are turned into a ProfileData: inlet molar flows, pressure, a temperature profile T(z),
and the observed mole fractions at the measured axial positions.

The reactor constants mirror scripts/methanation_on_NI_SIO/modeling/Simulation_Methods.py
in methanation-model, so the learned source terms have the same units and basis as Cantera's
surface net production rates for that reactor model (mol per m^2 of Ni per s).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# gas species carried in the reactor state; the reactive ones come first, N2 is inert
SPECIES = ['CH4', 'CO2', 'CO', 'H2', 'H2O', 'N2']
REACTIVE = ['CH4', 'CO2', 'CO', 'H2', 'H2O']

# names used by the mechanism / HDF5 profile columns
MECH_NAMES = {'CH4': 'CH4(1)', 'CO2': 'CO2(2)', 'CO': 'CO(5)', 'H2': 'H2(4)', 'H2O': 'H2O(3)', 'N2': 'N2'}

DEFAULT_DATA_ROOT = Path('/home/austin/methanation-model')

ONE_ATM = 101325.0  # Pa
R_GAS = 8.314462618  # J / mol / K


@dataclass(frozen=True)
class Reactor:
    """Profile reactor geometry and loading, as in Simulation_Methods.profile_simulation."""

    d_reactor: float = 4e-3  # m
    d_capillary: float = 0.7e-3  # m
    length: float = 45e-3  # m
    ghsv: float = 8850.0  # 1/h at STP
    ni_specific_area: float = 14.0  # m^2 Ni / g catalyst
    cat_mass: float = 0.4  # g
    porosity: float = 0.4
    n_cstr: int = 201  # Cantera's CSTR chain (only used to place the generated profile points)

    @property
    def area(self):
        return math.pi / 4.0 * (self.d_reactor ** 2 - self.d_capillary ** 2)

    @property
    def total_molar_flow(self):
        """mol/s. Cantera sets the mass flow from the STP volume flow at 273 K and 1 atm."""
        vdot = self.ghsv * self.area * self.length / 3600.0
        return ONE_ATM * vdot / (R_GAS * 273.0)

    @property
    def cat_area_per_length(self):
        """m^2 Ni per m of bed. Cantera scales the catalyst area by the porosity (rvol = A dz eps)."""
        return self.ni_specific_area * self.cat_mass * self.porosity / self.length


@dataclass
class ProfileData:
    names: list  # condition labels
    n_in: torch.Tensor  # (B, S) inlet molar flows [mol/s], order SPECIES
    P_atm: torch.Tensor  # (B,) pressure [atm]
    z_T: torch.Tensor  # (K,) positions of the temperature profile [m], shared
    T: torch.Tensor  # (B, K) temperature profile [K]
    z_obs: torch.Tensor  # (O,) observed positions [m], shared, increasing, > 0
    y_obs: torch.Tensor  # (B, O, S) observed mole fractions
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.names)

    def subset(self, idx):
        idx = list(idx)
        t = torch.as_tensor(idx, dtype=torch.long)
        return ProfileData([self.names[i] for i in idx], self.n_in[t], self.P_atm[t], self.z_T,
                           self.T[t], self.z_obs, self.y_obs[t], dict(self.meta))

    def to(self, device=None, dtype=None):
        conv = lambda x: x.to(device=device, dtype=dtype)
        return ProfileData(self.names, conv(self.n_in), conv(self.P_atm), conv(self.z_T), conv(self.T),
                           conv(self.z_obs), conv(self.y_obs), dict(self.meta))


def _as_tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _inlet_flows(x_in, reactor):
    x_in = np.asarray(x_in, dtype=np.float64)
    x_in = x_in / x_in.sum(axis=-1, keepdims=True)
    return x_in * reactor.total_molar_flow


def _read_profile(ds, key):
    """Mole fractions (201, S) of one Cantera CSTR-chain profile stored in the HDF5 files."""
    prof = pd.DataFrame(ds[:], columns=list(ds.attrs['columns']))
    y = prof[[MECH_NAMES[s] for s in SPECIES]].to_numpy()
    return y / y.sum(axis=1, keepdims=True)


def _cstr_positions(n_points, reactor):
    """Tank n's outlet sits at (n + 1) L/200 in plug-flow terms (Cantera writes it at n L/200)."""
    dz = reactor.length / (reactor.n_cstr - 1)
    return (np.arange(n_points) + 1) * dz


def _stride_keep(n_points, obs_stride):
    return np.unique(np.r_[np.arange(obs_stride - 1, n_points, obs_stride), n_points - 1])


def load_generated(data_root=DEFAULT_DATA_ROOT, reactor=Reactor(), mechanism=4758, obs_stride=1):
    """
    Profiles of the best-fit mechanism at the 200 generated conditions.

    The Cantera run feeds the composition in NiSiO2_generated_conditions.csv into a chain of
    201 CSTRs of length L/200 and records the outlet of tank n at 'Distance' n L/200. The
    outlet of tank n sits at (n + 1) L/200 in plug-flow terms, which is where it is placed here.
    """

    data_root = Path(data_root)
    cond = pd.read_csv(data_root / 'data' / 'NiSiO2_generated_conditions.csv')
    h5_file = data_root / 'results' / f'NiSiO2_generated_conditions_mechanism_{mechanism}.h5'

    import h5py

    names, x_in, P_atm, T, y_obs = [], [], [], [], []
    n_points = None

    with h5py.File(h5_file, 'r') as f:
        for _, row in cond.iterrows():
            key = f'condition_{int(row["condition"])}/mechanism_{mechanism}'
            if key not in f:  # failed simulations are missing from the file
                continue

            y = _read_profile(f[key], key)
            if n_points is None:
                n_points = len(y)
            elif len(y) != n_points:
                raise ValueError(f'{key} has {len(y)} points, expected {n_points}')
            y_obs.append(y)

            names.append(f'condition_{int(row["condition"])}')
            x_in.append(row[SPECIES].to_numpy(dtype=float))
            P_atm.append(row['p'] * 1e5 / ONE_ATM)
            T.append(float(row['Temp']))

    z_all = _cstr_positions(n_points, reactor)
    keep = _stride_keep(n_points, obs_stride)

    T = np.asarray(T)
    return ProfileData(
        names=names,
        n_in=_as_tensor(_inlet_flows(np.asarray(x_in), reactor)),
        P_atm=_as_tensor(P_atm),
        z_T=_as_tensor([0.0, z_all[-1]]),
        T=_as_tensor(np.stack([T, T], axis=1)),
        z_obs=_as_tensor(z_all[keep]),
        y_obs=_as_tensor(np.stack(y_obs)[:, keep, :]),
        meta={'dataset': 'generated', 'mechanism': mechanism, 'obs_stride': obs_stride},
    )


def load_experimental(data_root=DEFAULT_DATA_ROOT, reactor=Reactor(), n_experiments=24):
    """
    The 24 Küchen profiles. As in the Cantera simulations of these experiments, the measured
    composition at z = 0 is the inlet, and the measured temperature profile is imposed.
    """

    data_dir = Path(data_root) / 'data' / 'Kuchen_Experimental_data'

    names, x_in, P_atm, T, y_obs = [], [], [], [], []
    z_ref = None

    for n in range(n_experiments):
        df = pd.read_csv(data_dir / f'exp_{n}_data.csv').sort_values('z')
        z = df['z'].to_numpy(dtype=float) * 1e-3

        if z_ref is None:
            z_ref = z
        elif not np.allclose(z, z_ref):
            raise ValueError(f'exp_{n} is measured at different positions than exp_0')

        y = df[SPECIES].to_numpy(dtype=float)
        y = y / y.sum(axis=1, keepdims=True)

        names.append(f'exp_{n}')
        x_in.append(y[0])
        y_obs.append(y[1:])
        P_atm.append(df['p'].iloc[0] * 1e5 / ONE_ATM)
        T.append(df['Temp'].to_numpy(dtype=float))

    return ProfileData(
        names=names,
        n_in=_as_tensor(_inlet_flows(np.asarray(x_in), reactor)),
        P_atm=_as_tensor(P_atm),
        z_T=_as_tensor(z_ref),
        T=_as_tensor(np.stack(T)),
        z_obs=_as_tensor(z_ref[1:]),
        y_obs=_as_tensor(np.stack(y_obs)),
        meta={'dataset': 'experimental'},
    )


def load_experimental_sim(data_root=DEFAULT_DATA_ROOT, reactor=Reactor(), mechanism=4758, obs_stride=1,
                          n_experiments=24):
    """
    Mechanism 4758 simulated at the 24 Küchen conditions (simulate_5000_mechanisms_24_conditions.py).

    The simulation feeds the measured z = 0 composition into the 201-CSTR chain and sets each
    tank to the measured temperature interpolated at its position, so the inlet and T(z) here are
    taken from the experimental files exactly as in load_experimental. Only the observations
    differ: noise-free simulated profiles at 201 points instead of 23 measured points.
    """

    data_root = Path(data_root)
    exp_dir = data_root / 'data' / 'Kuchen_Experimental_data'
    h5_file = data_root / 'results' / 'NiSiO2_24_conditions.h5'

    import h5py

    names, x_in, P_atm, T, y_obs = [], [], [], [], []
    z_ref, n_points = None, None

    with h5py.File(h5_file, 'r') as f:
        for n in range(n_experiments):
            key = f'exp_{n}_on_NiSiO2/mechanism_{mechanism}'
            if key not in f:  # failed simulations are missing from the file
                print(f'Warning: {key} not in {h5_file.name}, skipping exp_{n}')
                continue

            y = _read_profile(f[key], key)
            if n_points is None:
                n_points = len(y)
            elif len(y) != n_points:
                raise ValueError(f'{key} has {len(y)} points, expected {n_points}')

            df = pd.read_csv(exp_dir / f'exp_{n}_data.csv').sort_values('z')
            z = df['z'].to_numpy(dtype=float) * 1e-3
            if z_ref is None:
                z_ref = z
            elif not np.allclose(z, z_ref):
                raise ValueError(f'exp_{n} is measured at different positions than exp_0')

            names.append(f'exp_{n}')
            x_in.append(df[SPECIES].to_numpy(dtype=float)[0])
            P_atm.append(df['p'].iloc[0] * 1e5 / ONE_ATM)
            T.append(df['Temp'].to_numpy(dtype=float))
            y_obs.append(y)

    keep = _stride_keep(n_points, obs_stride)
    return ProfileData(
        names=names,
        n_in=_as_tensor(_inlet_flows(np.asarray(x_in), reactor)),
        P_atm=_as_tensor(P_atm),
        z_T=_as_tensor(z_ref),
        T=_as_tensor(np.stack(T)),
        z_obs=_as_tensor(_cstr_positions(n_points, reactor)[keep]),
        y_obs=_as_tensor(np.stack(y_obs)[:, keep, :]),
        meta={'dataset': 'experimental_sim', 'mechanism': mechanism, 'obs_stride': obs_stride},
    )


DATASETS = ['generated', 'experimental', 'experimental_sim']


def load(dataset, data_root=DEFAULT_DATA_ROOT, reactor=Reactor(), obs_stride=1):
    """Load a dataset by name. obs_stride (every n-th axial point) only applies to the simulated ones."""
    if dataset == 'generated':
        return load_generated(data_root, reactor, obs_stride=obs_stride)
    if dataset == 'experimental':
        return load_experimental(data_root, reactor)
    if dataset == 'experimental_sim':
        return load_experimental_sim(data_root, reactor, obs_stride=obs_stride)
    raise ValueError(f'unknown dataset {dataset!r}')


def split_indices(n, fractions=(0.6, 0.2, 0.2), seed=0):
    """Random split of the conditions (not of the axial points) into train / val / test."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return (sorted(perm[:n_train].tolist()), sorted(perm[n_train:n_train + n_val].tolist()),
            sorted(perm[n_train + n_val:].tolist()))

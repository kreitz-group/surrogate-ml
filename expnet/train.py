"""
Sequential neural-ODE training of a +/- ExpNet on reactor profiles (paper Sec. 4.2):

    1. Euler with 1 control volume, then 2, 4, ..., max_cv control volumes, epochs_euler
       full-batch epochs each. After every stage the parameters with the lowest validation
       loss are restored before moving on.
    2. dopri8 (rtol 1e-6, atol 1e-10) for epochs_dopri epochs.

RMSprop, lr 1e-3, full batch. Loss: weighted MSE of mole fractions (Eq. 18) over all observed
axial points, which is the profile-reactor version of the paper's outlet-only loss.
"""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field

import pandas as pd
import torch

from .data import REACTIVE, SPECIES
from .model import ExpNet, count_parameters
from .reactor import PFR, mole_fractions, species_weights, weighted_mse


@dataclass
class Config:
    dataset: str = 'generated'
    data_root: str = '/home/austin/methanation-model'
    key_species: list = field(default_factory=lambda: ['CO', 'CH4'])
    loss_species: list = field(default_factory=lambda: ['CH4', 'CO2', 'CO', 'H2', 'H2O'])
    hidden: list = field(default_factory=lambda: [50])
    lr: float = 1e-3
    epochs_euler: int = 1000
    max_cv: int = 128
    epochs_dopri: int = 100
    rtol: float = 1e-6
    atol: float = 1e-10
    out_bias: float = None  # None: ln of a typical rate estimated from the training data
    split: list = field(default_factory=lambda: [0.6, 0.2, 0.2])
    split_seed: int = 0
    obs_stride: int = 1
    dtype: str = 'float64'
    device: str = 'cpu'
    log_every: int = 100


def input_ranges(train, reactor):
    """Min and max of [1/T, ln p_i] over the training inlets and observed profile points."""

    y = torch.cat([(train.n_in / train.n_in.sum(-1, keepdim=True)).unsqueeze(1), train.y_obs], dim=1)
    p = y[..., [SPECIES.index(s) for s in REACTIVE]] * train.P_atm.reshape(-1, 1, 1)
    ln_p = torch.log(p.clamp_min(1e-30)).reshape(-1, len(REACTIVE))
    inv_T = 1.0 / train.T.reshape(-1)

    x_min = torch.cat([inv_T.min().reshape(1), ln_p.min(0).values])
    x_max = torch.cat([inv_T.max().reshape(1), ln_p.max(0).values])

    # a constant input (e.g. a single temperature) would divide by zero in the min-max scaling
    flat = (x_max - x_min) < 1e-12
    x_min[flat] -= 0.5
    x_max[flat] += 0.5
    return x_min, x_max


def typical_rate(train, reactor):
    """ln of the mean |CO2 conversion| per Ni area over the training set, used as output bias."""
    i = SPECIES.index('CO2')
    dn = (train.n_in[:, i] - train.y_obs[:, -1, i] * train.n_in.sum(-1)).abs().mean()
    rate = dn / (reactor.cat_area_per_length * float(train.z_obs[-1]))
    return float(torch.log(rate.clamp_min(1e-30)))


def build_model(cfg, train, reactor, seed):
    torch.manual_seed(seed)
    x_min, x_max = input_ranges(train, reactor)
    out_bias = cfg.out_bias if cfg.out_bias is not None else typical_rate(train, reactor)
    net = ExpNet(REACTIVE, cfg.key_species, x_min, x_max, hidden=cfg.hidden, out_bias=out_bias)
    return PFR(net, reactor.cat_area_per_length)


def evaluate(pfr, data, weights, loss_species, method, n_cv=None, rtol=1e-6, atol=1e-10):
    n = pfr.solve(data, method=method, n_cv=n_cv, rtol=rtol, atol=atol)
    return weighted_mse(mole_fractions(n), data.y_obs, weights, loss_species)


def train_one(cfg, train, val, reactor, seed, out_dir=None, verbose=True):
    """Train one randomly initialized model through all stages. Returns (pfr, history DataFrame)."""

    dtype = getattr(torch, cfg.dtype)
    train = train.to(cfg.device, dtype)
    val = val.to(cfg.device, dtype)

    pfr = build_model(cfg, train, reactor, seed).to(cfg.device, dtype)
    weights = species_weights(train.y_obs, cfg.loss_species)

    stages = [('euler', 2 ** k, cfg.epochs_euler) for k in range(int(math.log2(cfg.max_cv)) + 1)]
    if cfg.epochs_dopri > 0:
        stages.append(('dopri8', None, cfg.epochs_dopri))

    # one optimizer for all stages: a freshly reset RMSprop takes a large first step that
    # throws away much of what the previous stage learned
    opt = torch.optim.RMSprop(pfr.parameters(), lr=cfg.lr)

    history = []
    t0 = time.perf_counter()

    for method, n_cv, epochs in stages:
        stage = f'{method}{n_cv or ""}'
        best_val, best_state = math.inf, copy.deepcopy(pfr.state_dict())

        for epoch in range(epochs):
            opt.zero_grad()
            try:
                loss = evaluate(pfr, train, weights, cfg.loss_species, method, n_cv, cfg.rtol, cfg.atol)
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite training loss')
                loss.backward()
                loss = float(loss.detach())
            except (AssertionError, RuntimeError, FloatingPointError) as e:
                # dopri8 can exceed max_num_steps if the current kinetics are very stiff; go back
                # to the best parameters of this stage and keep going
                if verbose:
                    print(f'  [seed {seed}] {stage} epoch {epoch}: {type(e).__name__}: {e}; restoring best')
                pfr.load_state_dict(best_state)
                continue
            opt.step()

            with torch.no_grad():
                try:
                    val_loss = float(evaluate(pfr, val, weights, cfg.loss_species, method, n_cv, cfg.rtol, cfg.atol))
                except (AssertionError, RuntimeError):
                    val_loss = math.inf

            if val_loss < best_val:
                best_val, best_state = val_loss, copy.deepcopy(pfr.state_dict())

            history.append({'stage': stage, 'epoch': epoch, 'train': loss, 'val': val_loss,
                            'time_s': time.perf_counter() - t0})

            if verbose and (epoch % cfg.log_every == 0 or epoch == epochs - 1):
                print(f'  [seed {seed}] {stage:>8s} epoch {epoch:5d}  train {loss:.3e}  '
                      f'val {val_loss:.3e}  best {best_val:.3e}  ({time.perf_counter() - t0:.0f} s)', flush=True)

        pfr.load_state_dict(best_state)

    history = pd.DataFrame(history)

    if out_dir is not None:
        save_model(pfr, cfg, out_dir / 'model.pt', seed=seed)
        history.to_csv(out_dir / 'history.csv', index=False)

    return pfr, history


def save_model(pfr, cfg, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    net = pfr.kinetics
    torch.save({
        'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
        'config': asdict(cfg),
        'reactive_species': net.reactive_species,
        'key_species': net.key_species,
        'hidden': list(cfg.hidden),
        'cat_area_per_length': pfr.a_cat,
        'n_parameters': count_parameters(net),
        **extra,
    }, path)


def load_model(path, device='cpu', dtype=torch.float64):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    net = ExpNet(ckpt['reactive_species'], ckpt['key_species'], sd['x_min'], sd['x_max'], hidden=ckpt['hidden'])
    net.load_state_dict(sd)
    pfr = PFR(net, ckpt['cat_area_per_length']).to(device, dtype)
    return pfr, ckpt


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

"""
Differentiable isothermal-per-position, isobaric plug-flow reactor driven by a +/- ExpNet
(the neural ODE of paper Sec. 4.2):

    d n_i / dz = a_cat * s_i(T(z), p(z))                                      (Eq. 17)

n_i  molar flow of species i [mol/s]
a_cat  Ni surface area per unit bed length [m^2/m]
s_i  source term from the ExpNet [mol/m^2/s]; zero for the inert N2

Two solvers are provided:
    euler(n_cv)  explicit Euler over n_cv equal control volumes. Used for the sequential
                 training stages 1, 2, 4, ..., 128 control volumes.
    dopri8       adaptive Dormand-Prince 8 from torchdiffeq, rtol 1e-6, atol 1e-10,
                 reverse-mode autodiff through the solver (no adjoint).

Negative molar flows are clipped at 1e-20 mol/s where the kinetics are evaluated.
"""

import torch

from .data import REACTIVE, SPECIES

N_FLOOR = 1e-20  # mol/s


def interp_rows(z, z_grid, values):
    """Linear interpolation of values (B, K) on the shared grid z_grid (K,) at the scalar z."""
    if z_grid.numel() == 1:
        return values[:, 0]
    k = torch.searchsorted(z_grid, z.reshape(1)).clamp(1, z_grid.numel() - 1)[0]
    w = ((z - z_grid[k - 1]) / (z_grid[k] - z_grid[k - 1])).clamp(0.0, 1.0)
    return values[:, k - 1] * (1.0 - w) + values[:, k] * w


class PFR(torch.nn.Module):

    def __init__(self, kinetics, cat_area_per_length):
        super().__init__()
        self.kinetics = kinetics
        self.a_cat = cat_area_per_length
        self.reactive_idx = [SPECIES.index(s) for s in REACTIVE]
        if list(kinetics.reactive_species) != REACTIVE:
            raise ValueError('ExpNet species must be in the order of data.REACTIVE')

        # set per call by solve()
        self._T = None
        self._z_T = None
        self._P = None
        self.n_evals = 0

    def source(self, n, T, P_atm):
        """Source terms of all species in SPECIES order [mol/m^2/s] at molar flows n (B, S)."""
        n_c = n.clamp_min(N_FLOOR)
        y = n_c / n_c.sum(dim=-1, keepdim=True)
        p = y[:, self.reactive_idx] * P_atm.unsqueeze(-1)
        s_reactive = self.kinetics(T, p)
        s = torch.zeros_like(n)
        s[:, self.reactive_idx] = s_reactive
        return s

    def forward(self, z, n):
        """RHS of the ODE: dn/dz [mol/s/m]."""
        self.n_evals += 1
        T = interp_rows(z, self._z_T, self._T)
        return self.a_cat * self.source(n, T, self._P)

    def solve(self, data, method='dopri8', n_cv=None, rtol=1e-6, atol=1e-10, max_num_steps=20000):
        """
        Integrate every condition in data from z = 0 and return the molar flows at data.z_obs,
        shape (B, O, S).
        """

        self._T, self._z_T, self._P = data.T, data.z_T, data.P_atm
        z_obs = data.z_obs

        if method == 'euler':
            return self._euler(data.n_in, z_obs, n_cv)

        if method in ('dopri8', 'dopri5'):
            from torchdiffeq import odeint
            t = torch.cat([z_obs.new_zeros(1), z_obs])
            sol = odeint(self, data.n_in, t, method=method, rtol=rtol, atol=atol,
                         options={'max_num_steps': max_num_steps})
            return sol[1:].transpose(0, 1)

        raise ValueError(f'unknown method {method!r}')

    def _euler(self, n0, z_obs, n_cv):
        """Explicit Euler over n_cv control volumes, linearly interpolated to z_obs."""

        length = z_obs[-1]
        h = length / n_cv
        z_grid = torch.arange(n_cv + 1, dtype=z_obs.dtype, device=z_obs.device) * h

        states = [n0]
        n = n0
        for k in range(n_cv):
            n = n + h * self(z_grid[k], n)
            states.append(n)
        states = torch.stack(states, dim=1)  # (B, n_cv + 1, S)

        k = torch.searchsorted(z_grid, z_obs).clamp(1, n_cv)
        w = ((z_obs - z_grid[k - 1]) / h).clamp(0.0, 1.0).reshape(1, -1, 1)
        return states[:, k - 1, :] * (1.0 - w) + states[:, k, :] * w


def mole_fractions(n):
    """Mole fractions from molar flows (..., S). Not clipped, so negative flows stay penalized."""
    return n / n.sum(dim=-1, keepdim=True)


def species_weights(y_obs_train, loss_species):
    """W_i = 1 / mean observed mole fraction in the training data (Eq. 20); zero for unused species."""
    w = torch.zeros(len(SPECIES), dtype=y_obs_train.dtype, device=y_obs_train.device)
    for s in loss_species:
        i = SPECIES.index(s)
        w[i] = 1.0 / y_obs_train[..., i].mean()
    return w


def weighted_mse(y_pred, y_obs, weights, loss_species):
    """Weighted mean squared error over conditions, axial points and the loss species (Eq. 18)."""
    return ((y_obs - y_pred) ** 2 * weights).sum(dim=-1).mean() / len(loss_species)

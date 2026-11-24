"""
+/- ExpNet (Exponential Scaling Formation Consumption Neural Network) and the atom
conservation layer, following Döppel, Kircher & Votsmeier, Chem. Eng. J. 535 (2026) 175101.

Layers (paper Fig. 2):
    1. inputs: temperature T [K] and partial pressures p_i [atm] of the reactive species
    2. latent linearization: x_T = 1/T, x_p,i = ln(p_i / 1 atm)                (Eqs. 2-3)
    3. clip to the training range, min-max scale to [-1, 1]                     (Eq. 4)
    4. hidden layer(s): z = tanh(W x + b)                                        (Eq. 5)
    5. linear layer to two nodes per key species: ln r_fwd, ln r_bwd            (Eq. 6)
    6. s_key = exp(ln r_fwd) - exp(ln r_bwd)                                     (Eq. 7)
    7. atom conservation layer: s = C . s_key                                    (Eq. 12)

C is the completion matrix: the reduced column echelon form of a null-space basis of the
atom-molecule matrix N, with the key species ordered first (paper Sec. 3.1, Table 4). Its
first n_key rows are the identity, so the unconstrained outputs are the key species source
terms and every other species follows from atom conservation.
"""

import numpy as np
import sympy
import torch
import torch.nn as nn

# elemental composition of every gas species that can appear in the reactor
ATOMS = {
    'CH4': {'C': 1, 'H': 4},
    'CO2': {'C': 1, 'O': 2},
    'CO': {'C': 1, 'O': 1},
    'H2': {'H': 2},
    'H2O': {'H': 2, 'O': 1},
    'N2': {'N': 2},
}


def completion_matrix(reactive_species, key_species):
    """
    Return the completion matrix C (len(reactive_species) x n_key) so that s = C @ s_key.

    The atom-molecule matrix N is built with the key species first, a basis of its null
    space is computed exactly with sympy, and that basis is brought into reduced column
    echelon form (rref of the transpose). The rows are then put back into the order of
    reactive_species.
    """

    if len(set(key_species)) != len(key_species) or not set(key_species) <= set(reactive_species):
        raise ValueError(f'key species {key_species} must be distinct members of {reactive_species}')

    order = list(key_species) + [s for s in reactive_species if s not in key_species]
    elements = sorted({e for s in order for e in ATOMS[s]})

    N = sympy.Matrix([[ATOMS[s].get(e, 0) for s in order] for e in elements])
    basis = N.nullspace()

    if len(basis) != len(key_species):
        raise ValueError(
            f'{len(order)} species with an atom matrix of rank {N.rank()} need exactly '
            f'{len(order) - N.rank()} key species, got {len(key_species)}')

    B = sympy.Matrix.hstack(*basis)
    C = B.T.rref()[0].T  # reduced column echelon form

    if C[:len(key_species), :] != sympy.eye(len(key_species)):
        raise ValueError(f'key species {key_species} are not independent under atom conservation')

    if N * C != sympy.zeros(N.rows, C.cols):
        raise RuntimeError('completion matrix does not conserve atoms')

    C = np.array(C.tolist(), dtype=float)

    # back to the order of reactive_species
    return C[[order.index(s) for s in reactive_species], :]


class ExpNet(nn.Module):
    """
    +/- ExpNet with the atom conservation layer.

    forward(T, p_atm) takes T with shape (B,) in K and p_atm with shape (B, n_reactive)
    in atm, and returns the source terms of the reactive species, shape (B, n_reactive),
    in the units the output bias was chosen for (mol / m^2_Ni / s in this project).
    """

    def __init__(self, reactive_species, key_species, x_min, x_max, hidden=(50,), out_bias=-10.0):
        super().__init__()

        self.reactive_species = list(reactive_species)
        self.key_species = list(key_species)
        self.n_key = len(key_species)

        C = completion_matrix(self.reactive_species, self.key_species)

        self.register_buffer('C', torch.as_tensor(C, dtype=torch.float64))
        self.register_buffer('x_min', torch.as_tensor(x_min, dtype=torch.float64))
        self.register_buffer('x_max', torch.as_tensor(x_max, dtype=torch.float64))

        n_inputs = 1 + len(self.reactive_species)
        if self.x_min.shape != (n_inputs,):
            raise ValueError(f'x_min must have {n_inputs} entries (1/T, then ln p of each reactive species)')

        layers = []
        width = n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.Tanh()]
            width = h
        self.hidden = nn.Sequential(*layers)

        # two nodes per key species: ln(formation rate) then ln(consumption rate)
        self.out = nn.Linear(width, 2 * self.n_key)

        # Xavier normal weights, zero hidden biases, output biases at a small rate (paper Sec. 4.2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.out.bias, out_bias)

    def features(self, T, p_atm, p_floor=1e-30):
        """Latent linearization (Eqs. 2-3): [1/T, ln p_1, ..., ln p_n]."""
        return torch.cat([(1.0 / T).unsqueeze(-1), torch.log(p_atm.clamp_min(p_floor))], dim=-1)

    def scale(self, x):
        """Clip to the training range, then min-max scale to [-1, 1] (Eq. 4)."""
        x = torch.maximum(torch.minimum(x, self.x_max), self.x_min)
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def latent_rates(self, T, p_atm):
        """Effective formation and consumption rates of the key species, each (B, n_key)."""
        ln_r = self.out(self.hidden(self.scale(self.features(T, p_atm))))
        return torch.exp(ln_r[..., :self.n_key]), torch.exp(ln_r[..., self.n_key:])

    def forward(self, T, p_atm):
        r_fwd, r_bwd = self.latent_rates(T, p_atm)
        s_key = r_fwd - r_bwd
        return s_key @ self.C.T


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

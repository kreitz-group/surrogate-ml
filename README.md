# surrogate-ml

Repo for the generation of the kinetic data for a surrogate microkinetic model using ML.

Steady-state surface coverages and reaction rates for CO2 methanation are computed with Cantera across a Sobol sampled space of operating conditions, with gas-phase compositions drawn from a Dirichlet distribution.

The repo also contains a +/- ExpNet neural kinetic model (Döppel et al. 2026) trained directly on Ni/SiO2 profile-reactor data from the `methanation-model` repo. See [Neural kinetics from reactor profiles](#neural-kinetics-from-reactor-profiles---expnet).

## Repository overview

```text
surrogate-ml/
|-- data/
|   `-- mechanism_files/
|       `-- File_303/
|           |-- chem_covdep.yaml            original coverage-dependent mechanism
|           `-- chem_covdep_formatted.yaml  same mechanism, formatted for Cantera 4.0 (used by the scripts)
|-- results/
|   |-- sobol_dirichlet/
|   |   |-- sobol_points.csv                sampled operating conditions
|   |   |-- sobol_parameter_space_0.3.png   temperature vs. each mole fraction, plus H2 vs. CO2
|   |   `-- sobol_purity.png                histogram of the largest mole fraction in each mixture
|   |-- ML_data/
|   |   `-- rates_coverages.csv             simulation output / ML training data
|   `-- expnet/                             trained +/- ExpNet runs (see below)
|-- expnet/                                 +/- ExpNet package (model, reactor, data, training)
|-- scripts/
|   |-- generate_sobol_dirichlet.py
|   |-- calc_rates_coverages.py
|   |-- train_expnet.py
|   |-- evaluate_expnet.py
|   `-- expnet_cantera_source_terms.py
|-- environment.yaml
|-- LICENSE
`-- README.md
```

All paths in the scripts are resolved relative to the repository root, so the scripts can be run from any directory. The `results/` subdirectories are created by the scripts when they are run.

## Environment setup

The scripts are written in Python and rely on Cantera. The recommended way to reproduce the software environment for this repository is with the provided Conda environment file.

Create the environment with:

```bash
conda env create -f environment.yaml
```

Then activate it:

```bash
conda activate ct-4.0
```

The environment name is defined by the `name:` field inside `environment.yaml`.

This project uses a developer version of Cantera: Cantera 4.0. After creating and activating the Conda environment, Cantera should be compiled from source and installed into that environment by following the official Cantera documentation:

- [Cantera installation guide](https://cantera.org/dev/develop/index.html)

## Workflow

### 1. Generate the operating conditions

```bash
python scripts/generate_sobol_dirichlet.py
```

This generates 2^16 = 65,536 points of an 8-dimensional Sobol sequence (skipping the first 500 points). The first dimension is scaled to the temperature bounds, and the remaining seven dimensions are mapped to the gas-phase mole fractions:

| Parameter | Range |
|---|---|
| Temperature | 220 – 300 °C |
| Mole fractions of H2, CO2, CH4, CO, H2O, C2H6, Ar | 0 – 1, summing to 1 |

The mole fractions are sampled from a symmetric Dirichlet distribution with concentration parameter alpha = 0.3. Each of the seven uniform Sobol coordinates is passed through the inverse CDF of a Gamma(alpha) distribution, and the resulting gamma variables are normalized by their sum, which gives a Dirichlet(alpha, ..., alpha) sample on the composition simplex. With alpha < 1 the samples are pushed toward the edges and corners of the simplex, so the data set covers both dilute mixtures and mixtures dominated by one or a few species rather than clustering around an equal-parts composition.

The points are written to `results/sobol_dirichlet/sobol_points.csv` (columns `Temperature`, `H2`, `CO2`, `CH4`, `CO`, `H2O`, `C2H6`, `Ar`), along with a plot of the sampled parameter space and a histogram of the largest mole fraction in each mixture.

### 2. Run the microkinetic simulations

```bash
python scripts/calc_rates_coverages.py
```

For each sampled point, the surface phase of mechanism `File_303` (`chem_covdep_formatted.yaml`) is set to the given temperature and gas composition at 3 bar and advanced to steady-state coverages. Conditions are run in parallel with a process pool; each worker loads the mechanism once and resets the surface to its initial coverages before every condition, which keeps memory use bounded. Each row is written to `results/ML_data/rates_coverages.csv` as it finishes.

Conditions where the steady-state solver fails are skipped, and the number of failed conditions is printed at the end of the run.

### Output columns

Each row of `rates_coverages.csv` contains:

- `Condition` — row index of the point in `sobol_points.csv` (skipped conditions leave gaps)
- `T (C)`, `P (bar)` — temperature and pressure
- `x_H2`, `x_CO2`, `x_CH4`, `x_CO`, `x_H2O`, `x_C2H6`, `x_Ar` — gas-phase mole fractions
- one column per surface species (e.g. `HX(8)`, `CO2X(9)`) — steady-state coverages
- `R<n>: <equation>` — net rate of progress of each reaction (kmol/m^2/s)
- `prod_<species>` — net production rate of each gas and surface species (kmol/m^2/s)

## Neural kinetics from reactor profiles (+/- ExpNet)

`expnet/` implements the Exponential Scaling Formation Consumption Neural Network (+/- ExpNet) of Döppel, Kircher & Votsmeier, *Chem. Eng. J.* 535 (2026) 175101. The network learns CO2 methanation kinetics directly from the Ni/SiO2 profile-reactor data in the `methanation-model` repo. It is trained as a neural ODE: the network supplies the source terms of a plug-flow reactor model, and the reactor is integrated and compared with the measured or simulated mole-fraction profiles.

### How it fits together

```text
methanation-model/                         surrogate-ml/
  results/NiSiO2_generated_..._4758.h5  ─┐
  data/NiSiO2_generated_conditions.csv  ─┼─> expnet/data.py ─> train_expnet.py ─> results/expnet/<dataset>/<tag>/
  data/Kuchen_Experimental_data/*.csv   ─┘                                             │
                                                                 evaluate_expnet.py <──┘
```

| File | Purpose |
|---|---|
| `expnet/model.py` | The +/- ExpNet and the atom conservation (completion) matrix |
| `expnet/reactor.py` | Differentiable plug-flow reactor (Euler and dopri8 solvers), loss function |
| `expnet/data.py` | Reactor constants and the loaders for both datasets |
| `expnet/train.py` | Staged training loop, saving and loading models |
| `scripts/train_expnet.py` | Command-line training, several seeds in parallel |
| `scripts/evaluate_expnet.py` | Metrics and plots for a trained run |
| `scripts/expnet_cantera_source_terms.py` | Mechanism 4758's true source terms, used to check the learned kinetics (uses Cantera) |

The scripts only **read** from `methanation-model`, which defaults to `/home/austin/methanation-model`. On another machine, pass `--data-root /path/to/methanation-model`.

### Method

- **Network** (paper Fig. 2): the inputs are T and the partial pressures of CH4, CO2, CO, H2 and H2O. They are transformed inside the network to 1/T and ln(p/1 atm), clipped to the training range and min-max scaled to [-1, 1]. A tanh hidden layer (50 nodes) follows, then two linear nodes per key species. These nodes are ln(formation rate) and ln(consumption rate), and the key species source term is `exp(ln r_fwd) - exp(ln r_bwd)`, so it can change sign.
- **Atom conservation layer:** `s = C · s_key`. C is the reduced column echelon form of the null space of the C/H/O atom-molecule matrix, with the key species **CO and CH4** first. The other species follow from atom conservation: `s_CO2 = -s_CO - s_CH4`, `s_H2 = -s_CO - 4 s_CH4`, `s_H2O = s_CO + 2 s_CH4`. N2 is inert. The model has 554 trainable parameters.
- **Reactor model:** an isobaric PFR, `dṅ_i/dz = a_cat · s_i(T(z), p(z))`. The geometry, GHSV, catalyst loading and porosity are copied from `methanation-model/.../Simulation_Methods.py`, so `s` is in mol per m² Ni per s, on the same basis as Cantera's surface `net_production_rates` for that reactor. T(z) is constant for the generated data and the measured profile for the Küchen data.
- **Training** (paper Sec. 4.2): full batch, RMSprop with lr 1e-3. Explicit Euler is run with 1, 2, 4, ..., 128 control volumes for 1000 epochs each, and the best-validation parameters are restored after every stage. A final stage of 100 epochs uses dopri8 (torchdiffeq, rtol 1e-6, atol 1e-10, backpropagation through the solver). The loss is the weighted MSE of the mole fractions, with weights equal to 1 / the mean training mole fraction. **Because this is a profile reactor, the loss uses every measured axial point, not only the outlet.** Conditions are split 60/20/20 into train/val/test. 10 Xavier-normal initializations are trained, and the one with the lowest validation loss is kept.

Differences from the paper are deliberate and small: (1) a single RMSprop optimizer is kept for all stages, because resetting it caused a loss spike at each stage change; (2) the output bias starts at ln of a typical rate estimated from the training data instead of -10, because the units differ; (3) the loss uses the profile points as described above.

### Datasets (`--dataset`)

| name | source (in `methanation-model`) | conditions | points per profile |
|---|---|---|---|
| `generated` (default) | `results/NiSiO2_generated_conditions_mechanism_4758.h5` + `data/NiSiO2_generated_conditions.csv` | 200 (isothermal, simulated with mechanism 4758) | 201 |
| `experimental` | `data/Kuchen_Experimental_data/exp_*_data.csv` | 24 (measured, with T profile) | 23 after the z = 0 inlet |

- For the generated data, Cantera's 201-CSTR chain outputs tank *n* at `Distance = n·L/200`. In plug-flow terms this point is at (n+1)·L/200, and that is where it is placed. The CSV composition is the inlet.
- For the experimental data, the measured composition at z = 0 is the inlet, as in the Cantera simulations of those experiments. H2O in those files is calculated as CO + 2·CH4, not measured, so its error is not independent of the CH4 and CO errors.

### Setup

All commands are run from the repository root with the environment active:

```bash
conda activate ct-4.0
```

Training needs `torch` and `torchdiffeq`. Cantera is only needed for `expnet_cantera_source_terms.py`. If `torchdiffeq` is missing, install it without letting pip change the torch build:

```bash
pip install --no-deps torchdiffeq
```

### 1. Smoke test (about 10 seconds)

Run this first on any new machine to check the install and the data paths. The accuracy will be poor; it only checks that everything works.

```bash
python scripts/train_expnet.py --quick                          # generated data
python scripts/train_expnet.py --quick --dataset experimental
```

Each run writes to `results/expnet/<dataset>/quick/`.

### 2. Train

On a laptop (one model, about 5-7 min):

```bash
python scripts/train_expnet.py --dataset generated --seeds 1 --threads 2 --tag mytest
```

On the server (the paper's protocol: 10 models with different random starts, best one kept):

```bash
python scripts/train_expnet.py --dataset generated    --seeds 10 --workers 10 --threads 1 --tag paper --data-root /path/to/methanation-model
python scripts/train_expnet.py --dataset experimental --seeds 10 --workers 10 --threads 1 --tag paper --data-root /path/to/methanation-model
```

- `--workers` is how many models train at the same time. Each one uses `--threads` CPU threads, so keep `workers × threads` at or below the number of cores.
- **Use CPU, not the GPU.** The network is tiny (554 parameters, about 120 conditions per batch), so per-step overhead dominates, and float64 is slow on consumer GPUs. `--device cuda` exists but will likely be slower. One seed with the paper settings took 7 min (generated) and 5.5 min (experimental) on the laptop with 2 threads.
- For long runs, use `nohup python scripts/train_expnet.py ... > train.log 2>&1 &` or `tmux` so they survive logging out.
- On the server: copy or clone both repos, create the env from `environment.yaml`, and check that `torchdiffeq` was installed.

Each model prints a progress line every `--log-every` epochs:

```text
[seed 0]  euler16 epoch   500  train 3.652e-05  val 4.081e-05  best 3.997e-05  (69 s)
```

`euler1` ... `euler128` are the warm-up stages with a coarse solver, and `dopri8` is the final stage with an accurate adaptive solver. `val` is the error on conditions held out from training; it should keep going down.

Outputs in `results/expnet/<dataset>/<tag>/`:

| File | Contents |
|---|---|
| `config.json` | Every setting used for the run |
| `split.json` | Which conditions are in train, val and test |
| `seed_<k>/model.pt`, `seed_<k>/history.csv` | Each trained model and its loss curve |
| `summary.csv` | Final train and validation loss of every model, best first |
| `best_model.pt` | The model to use |

**Use a new `--tag` for each run.** Reusing a tag overwrites that folder.

### 3. Evaluate

```bash
python scripts/evaluate_expnet.py results/expnet/generated/paper
python scripts/evaluate_expnet.py results/expnet/generated/paper --show    # also opens the plots
python scripts/evaluate_expnet.py results/expnet/generated/paper --profiles test   # profile plots for test conditions only
```

This integrates the best model with dopri8 for every condition and writes to `<run>/eval/`:

| File | What to look at |
|---|---|
| `metrics.csv` | Error per split (train/val/test) and per species. **Judge the model on the test rows**, which it never saw. |
| `parity_outlet.png` | Predicted vs. observed outlet mole fraction. Dashed lines mark ±20%. |
| `profiles/<condition>.png` | One plot per condition (e.g. `profiles/exp_14.png`): ExpNet model (lines) vs. experiment or Cantera data (circles), with the split in the title. All conditions by default; `--profiles test` (or `train`/`val`) limits it. The folder is cleared on each run. |
| `history.png` | Loss curve through all the stages |
| `predictions.csv` | Observed and predicted values at every axial point |

### 4. Check the learned rates against Cantera (optional, uses Cantera)

The model only ever sees mole fractions. This step checks whether the rates it learned match mechanism 4758's true rates at the same conditions:

```bash
python scripts/expnet_cantera_source_terms.py --dataset generated --stride 10
python scripts/evaluate_expnet.py results/expnet/generated/paper --cantera results/expnet/cantera_source_terms_generated.csv
```

- `--stride 10` uses every 10th axial point.
- For the generated data it reuses the steady-state coverages stored in the HDF5 file, so no solve is needed and it should be fast. Pass `--resolve` to solve the coverages again instead.
- `--dataset experimental` solves the surface to steady state at every measured point, so it is slower and some points may fail to converge.
- The comparison is written to `eval/source_terms.png` and `eval/source_term_metrics.csv`.

### 5. Use a trained model in your own code

```python
import sys, torch
sys.path.insert(0, '/home/austin/surrogate-ml')
from expnet import data as D
from expnet.train import load_model
from expnet.reactor import mole_fractions

pfr, ckpt = load_model('results/expnet/generated/paper/best_model.pt')
```

**(a) Rates at given conditions.** This is the fast replacement for a Cantera steady-state solve, and it accepts thousands of rows at once:

```python
T = torch.tensor([623.15], dtype=torch.float64)                          # K
x = torch.tensor([[0.01, 0.08, 0.002, 0.35, 0.02]], dtype=torch.float64)  # CH4, CO2, CO, H2, H2O
P_atm = 3.0 / 1.01325
with torch.no_grad():
    s = pfr.kinetics(T, x * P_atm)    # mol / m^2 Ni / s, same order as D.REACTIVE
```

**(b) Simulate the reactor for a new inlet:**

```python
reactor = D.Reactor()
y_in = torch.tensor([[1e-6, 0.10, 1e-6, 0.40, 1e-6, 0.50]], dtype=torch.float64)  # CH4, CO2, CO, H2, H2O, N2
z = torch.linspace(0.5e-3, 45e-3, 90, dtype=torch.float64)                       # m
new = D.ProfileData(names=['my_case'], n_in=y_in / y_in.sum() * reactor.total_molar_flow,
                    P_atm=torch.tensor([3.0 / 1.01325], dtype=torch.float64),
                    z_T=torch.tensor([0.0, 45e-3], dtype=torch.float64),
                    T=torch.tensor([[573.15, 573.15]], dtype=torch.float64),    # T at each z_T point (K)
                    z_obs=z, y_obs=torch.zeros(1, len(z), 6, dtype=torch.float64))
with torch.no_grad():
    y = mole_fractions(pfr.solve(new, method='dopri8'))   # (1, 90, 6) mole fractions along z
```

For a non-uniform temperature, put more points in `z_T` and `T`. **Stay inside the training range** (roughly 235-360 °C, 2-5 bar, and the feed compositions of the conditions file). Outside it the inputs are clipped to the training limits, so the results are unreliable.

### Settings worth changing

| Flag | Default | When to change |
|---|---|---|
| `--epochs-euler`, `--epochs-dopri` | 1000, 100 | Raise these if `history.png` is still going down at the end of a stage |
| `--lr` | 1e-3 | Try 3e-4 if the final dopri8 stage is noisy |
| `--hidden` | `50` | Try `--hidden 100` or `--hidden 50 50` for a bigger model |
| `--key-species` | `CO CH4` | The paper recommends the low-concentration species, so leave this unless experimenting |
| `--split-seed` | 0 | Changes which conditions are held out. Useful for the experimental set, which has only 5 test experiments. |
| `--obs-stride` | 1 | Use every n-th axial point of the generated profiles |
| `--tag` | timestamp | Name of the run folder |

`python scripts/train_expnet.py --help` lists every option.

### Results so far (laptop, 1 seed)

Mean relative error of the outlet mole fraction on the test conditions:

| Dataset | Test conditions | CH4 | CO2 | CO | H2 | H2O |
|---|---|---|---|---|---|---|
| generated | 40 | 4.0% | 1.1% | 8.4% | 0.9% | 3.7% |
| experimental | 5 | 9.3% | 9.6% | 21% | 10% | 9.0% |

These are from `results/expnet/<dataset>/laptop_1seed/`. In the generated run the dopri8 stage was still noisy, so more epochs or a lower learning rate may help. The experimental model underpredicts the highest-conversion experiments (exp_14, exp_15) and has only 14 training experiments.

### Troubleshooting

- **`FileNotFoundError` for the `.h5` or CSV files:** `--data-root` does not point at `methanation-model`.
- **`ModuleNotFoundError: torchdiffeq`:** run `pip install --no-deps torchdiffeq`.
- **`... max_num_steps ...; restoring best` during dopri8:** the solver hit its step limit for that epoch. The run goes back to the best parameters of the stage and carries on. A few of these are harmless; constant ones mean the learning rate is too high.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

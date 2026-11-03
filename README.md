# surrogate-ml

Repo for the generation of the kinetic data for a surrogate microkinetic model using ML.

Steady-state surface coverages and reaction rates for CO2 methanation are computed with Cantera across a Sobol sampled space of operating conditions, with gas-phase compositions drawn from a Dirichlet distribution.

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
|   `-- ML_data/
|       `-- rates_coverages.csv             simulation output / ML training data
|-- scripts/
|   |-- generate_sobol_dirichlet.py
|   `-- calc_rates_coverages.py
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

## License

This project is licensed under the MIT License. See `LICENSE` for details.

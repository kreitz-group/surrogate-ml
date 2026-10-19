# surrogate-ml

Repo for the generation of the kinetic data for a surrogate microkinetic model using ML.

Steady-state surface coverages and reaction rates for CO2 methanation are computed with Cantera across a Sobol sampled space of operating conditions.

## Repository overview

```text
surrogate-ml/
|-- data/
|   `-- mechanism_files/
|       `-- File_303/
|           |-- chem_covdep.yaml            original coverage-dependent mechanism
|           `-- chem_covdep_formatted.yaml  same mechanism, formatted for Cantera 4.0 (used by the scripts)
|-- results/
|   |-- sobol_sequence/
|   |   |-- sobol_points.csv                sampled operating conditions
|   |   `-- sobol_parameter_space.png       pairwise plots of the sampled conditions
|   `-- ML_data/
|       `-- rates_coverages.csv             simulation output / ML training data
|-- scripts/
|   |-- generate_sobol.py
|   `-- calc_rates_coverages.py
|-- environment.yaml
|-- LICENSE
`-- README.md
```

All paths in the scripts are resolved relative to the repository root, so the scripts can be run from any directory.

The `results/` directory is not tracked by git (see `.gitignore`). It is created by the scripts when they are run, so the outputs have to be regenerated locally by following the workflow below.

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
python scripts/generate_sobol.py
```

This generates 2^14 = 16,384 points of a 3-dimensional Sobol sequence (skipping the first 500 points) and scales them to the parameter bounds:

| Parameter | Range |
|---|---|
| Temperature | 220 – 300 °C |
| H2/CO2 ratio | 0.5 – 6 |
| Ar mole fraction | 0 – 0.75 |

The points are written to `results/sobol_sequence/sobol_points.csv` along with a plot of the sampled parameter space.

### 2. Run the microkinetic simulations

```bash
python scripts/calc_rates_coverages.py
```

For each Sobol point, the surface phase of mechanism `File_303` (`chem_covdep_formatted.yaml`) is set to the given temperature and gas composition at 3 bar and advanced to steady-state coverages. Conditions are run in parallel with a process pool, and each row is written to `results/ML_data/rates_coverages.csv` as it finishes.

The gas composition is computed from the H2/CO2 ratio and Ar fraction as:

- x_CO2 = (1 - x_Ar) / (1 + H2/CO2)
- x_H2 = 1 - x_Ar - x_CO2

### Output columns

Each row of `rates_coverages.csv` contains:

- `Condition` — row index of the Sobol point
- `T (C)`, `P (bar)`, `H2_CO2_ratio`, `x_Ar` — operating conditions
- one column per surface species (e.g. `HX(8)`, `CO2X(9)`) — steady-state coverages
- `R<n>: <equation>` — net rate of progress of each reaction (kmol/m^2/s)
- `prod_<species>` — net production rate of each gas and surface species (kmol/m^2/s)

## License

This project is licensed under the MIT License. See `LICENSE` for details.

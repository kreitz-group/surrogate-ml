import cantera as ct
import numpy as np
import pandas as pd

import csv
import sys

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# each worker process loads the mechanism once and reuses it for every condition,
# since creating a new Interface for each condition leaks memory

surf1 = None
initial_coverages = None


def init_worker(yaml_file: Path):

    global surf1, initial_coverages

    surf1 = ct.Interface(yaml_file, name='surface1')
    initial_coverages = surf1.coverages.copy()


def main():

    # Use mechanism 303

    yaml_dir = REPO_ROOT / 'data' / 'mechanism_files' / 'File_303' / 'chem_covdep_formatted.yaml'

    # load in the sobol generated paramters

    sobol_dir = REPO_ROOT / 'results' / 'sobol_dirichlet' / 'sobol_points.csv'
    sobol_df = pd.read_csv(sobol_dir)

    condition_numbers = list(sobol_df.index)
    conditions = [serie for _, serie in sobol_df.iterrows()]

    out_dir = REPO_ROOT / 'results' / 'ML_data'
    out_dir.mkdir(parents=True, exist_ok=True)

    # write each run to the csv as it finishes

    with ProcessPoolExecutor(initializer=init_worker, initargs=(yaml_dir,)) as executor, \
         (out_dir / 'rates_coverages.csv').open('w', newline='') as csv_file:

        writer = csv.writer(csv_file)

        results = executor.map(compute_rate_coverage, conditions)

        # skip conditions that failed to converge, and write the headers with the first one that didn't

        header_written = False
        n_failed = 0

        for condition_number, result in tqdm(zip(condition_numbers, results),
                                             total=len(condition_numbers)):
            if result is None:
                n_failed += 1
                continue

            headers, data = result

            if not header_written:
                writer.writerow(['Condition'] + headers)
                header_written = True

            writer.writerow([condition_number] + data)

    print(n_failed)


def compute_rate_coverage(conditions: pd.Series):

    # unit conversion factors to SI

    bar = 1e5

    p_bar = 3 # pressure in bar

    #parse through the experimental condtions in the pandas series

    tc = float(conditions['Temperature']) # Temperature in C

    t = tc + 273.15

    # read the mole fractions

    x_H2 = float(conditions['H2'])
    x_CO2 = float(conditions['CO2'])
    x_CH4 = float(conditions['CH4'])
    x_CO = float(conditions['CO'])
    x_H2O = float(conditions['H2O'])
    x_C2H6 = float(conditions['C2H6'])
    x_Ar = float(conditions['Ar'])

    # put the mole fractions into a dictionary

    composition_dict = {
        'H2(4)': x_H2,
        'CO2(2)': x_CO2,
        'CH4(1)': x_CH4,
        'CO(5)': x_CO,
        'H2O(3)': x_H2O,
        'C2H6(6)': x_C2H6,
        'Ar': x_Ar,
    }

    p = p_bar * bar # pressure

    # reset this worker's surface to its initial coverages and set the new conditions

    surf1.TP = t, p
    surf1.coverages = initial_coverages

    gas1 = surf1.adjacent['gas']
    gas1.TPX = t, p, composition_dict

    # return None if the steady state solver fails so the condition is skipped

    try:
        surf1.advance_coverages_to_steady_state()
    except ct.CanteraError:
        return None

    # label the rates with the reaction number so every column name is unique

    rate_names = [f'R{j + 1}: {eq}' for j, eq in enumerate(surf1.reaction_equations())]

    production_names = [f'prod_{surf1.kinetics_species_name(k)}' for k in range(surf1.n_total_species)]

    headers = (['T (C)', 'P (bar)', 'x_H2', 'x_CO2', 'x_CH4', 'x_CO', 'x_H2O', 'x_C2H6', 'x_Ar']
               + surf1.species_names
               + rate_names
               + production_names)

    # record the gas mole fractions, surface coverages, net rates of progress (kmol/m^2/s),
    # and net production rates of every species (kmol/m^2/s)

    output_data = (
        [surf1.T - 273.15, surf1.P / (10 ** 5), x_H2, x_CO2, x_CH4, x_CO, x_H2O, x_C2H6, x_Ar]
        + list(surf1.coverages)
        + list(surf1.net_rates_of_progress)
        + list(surf1.net_production_rates)
    )

    return headers, output_data


if __name__ == "__main__":
    main()
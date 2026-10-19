import cantera as ct
import numpy as np
import pandas as pd

import csv
import sys

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():

    # Use mechanism 303

    yaml_dir = REPO_ROOT / 'data' / 'mechanism_files' / 'File_303' / 'chem_covdep_formatted.yaml'

    # load in the sobol generated paramters

    sobol_dir = REPO_ROOT / 'results' / 'sobol_sequence' / 'sobol_points.csv'
    sobol_df = pd.read_csv(sobol_dir)

    condition_numbers = list(sobol_df.index)
    conditions = [serie for _, serie in sobol_df.iterrows()]

    out_dir = REPO_ROOT / 'results' / 'ML_data'
    out_dir.mkdir(parents=True, exist_ok=True)

    # write each run to the csv as it finishes,
    # restarting each worker after 200 conditions so its memory is returned to the OS

    with ProcessPoolExecutor(max_tasks_per_child=200) as executor, \
         (out_dir / 'rates_coverages.csv').open('w', newline='') as csv_file:

        writer = csv.writer(csv_file)

        results = executor.map(compute_rate_coverage,
                               [yaml_dir] * len(condition_numbers),
                               conditions)

        for condition_number, (headers, data) in tqdm(zip(condition_numbers, results),
                                                      total=len(condition_numbers)):
            if condition_number == condition_numbers[0]:
                writer.writerow(['Condition'] + headers)

            writer.writerow([condition_number] + data)


def compute_rate_coverage(yaml_file: Path, conditions: pd.Series):

    # unit conversion factors to SI

    bar = 1e5

    p_bar = 3 # pressure in bar

    #parse through the experimental condtions in the pandas series

    tc = float(conditions['Temperature']) # Temperature in C

    t = tc + 273.15

    # calculate the mole fractions

    ratio = float(conditions['H2_CO2_ratio'])

    x_Ar = float(conditions[' %Ar'])
    x_CO2 = (1-x_Ar) / (1 + ratio)
    x_H2 = 1 - x_Ar - x_CO2

    # put the mole fractions into a dictionary

    composition_dict = {
        'Ar': x_Ar,
        'CO2(2)': x_CO2,
        'H2(4)': x_H2,
    }

    p = p_bar * bar # pressure

    surf1 = ct.Interface(yaml_file, name='surface1')
    surf1.TP = t, p

    gas1 = surf1.adjacent['gas']
    gas1.TPX = t, p, composition_dict

    surf1.advance_coverages_to_steady_state()

    # label the rates with the reaction number so every column name is unique

    rate_names = [f'R{j + 1}: {eq}' for j, eq in enumerate(surf1.reaction_equations())]

    production_names = [f'prod_{surf1.kinetics_species_name(k)}' for k in range(surf1.n_total_species)]

    headers = (['T (C)', 'P (bar)', 'H2_CO2_ratio', 'x_Ar']
               + surf1.species_names
               + rate_names
               + production_names)

    # record the gas mole fractions, surface coverages, net rates of progress (kmol/m^2/s),
    # and net production rates of every species (kmol/m^2/s)

    output_data = (
        [surf1.T - 273.15, surf1.P / (10 ** 5), ratio, x_Ar]
        + list(surf1.coverages)
        + list(surf1.net_rates_of_progress)
        + list(surf1.net_production_rates)
    )

    return headers, output_data


if __name__ == "__main__":
    main()
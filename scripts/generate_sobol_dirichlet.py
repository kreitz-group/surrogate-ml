import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import optunity
from scipy.stats import gamma
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#Makes the diagrams look nice and shiny
plt.rcParams['figure.figsize']=(14,5)
plt.rcParams['axes.linewidth'] = 2
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rcParams['lines.markersize'] = 10
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams['xtick.major.size']=10
plt.rcParams['xtick.major.width']=2
plt.rcParams['ytick.major.size']=10
plt.rcParams['ytick.major.width']=2
plt.rcParams['legend.edgecolor']='k'
plt.rcParams['axes.unicode_minus']=False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams['legend.handletextpad']=0.4
plt.rcParams['legend.columnspacing']=0.5
plt.rcParams['legend.labelspacing']=0.3
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['axes.formatter.limits']=(-3, 6)

num_pts = 2 ** 16   # the number of points to generate
skip = 500    # the number of initial points of the Sobol sequence to skip

# Dirichlet concentration parameter

alpha = 0.3

#One dimension each for: Temperature, H2, CO2, CH4, CO, H2O, C2H6, Ar

N = 8 # number of dimensions

species = ['H2', 'CO2', 'CH4', 'CO', 'H2O', 'C2H6', 'Ar']
species_labels = [r'H$_2$', r'CO$_2$', r'CH$_4$', 'CO', r'H$_2$O', r'C$_2$H$_6$', 'Ar']

out_dir = REPO_ROOT / 'results' / 'sobol_dirichlet'
out_dir.mkdir(parents=True, exist_ok=True)

# paramter bounds

temp = [220, 300] # temperature bounds in celcius

# generate Sobol sequence

res = optunity.solvers.Sobol.i4_sobol_generate(N, num_pts, skip)
parameters = np.array(res)

# scale the the temperature to the bounds

parameters[:,0] = temp[0] + parameters[:,0] * (temp[1] - temp[0])

# map the mole fractions onto the simplex with a Dirichlet(alpha,...,alpha) distribution
# the inverse CDF of Gamma(alpha) turns each uniform into a gamma variable,
# and normalized gamma variables are Dirichlet(alpha)

cols = slice(1,8)

u = np.clip(parameters[:,cols], 1e-12, 1 - 1e-12) 
g = gamma.ppf(u, a=alpha)
parameters[:,cols] = g / np.sum(g, axis=1, keepdims=True)

print('max deviation of row sums from 1:', np.abs(parameters[:,cols].sum(axis=1) - 1).max())
print('points with one species > 90%:', np.mean(parameters[:,cols].max(axis=1) > 0.9) * 100, '%')

#Save the file

df = pd.DataFrame(parameters, columns=['Temperature'] + species)
df.to_csv(out_dir / 'sobol_points.csv', index=False, float_format='%.6e')

# temperature vs. each species mole fraction, plus H2 vs. CO2

fig = plt.figure(figsize=(24, 10))
gs=gridspec.GridSpec(nrows=2, ncols=4)
gs.update(wspace=0.35, hspace=0.3)

for i, label in enumerate(species_labels):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.plot(parameters[:,0],parameters[:,i+1] ,  color='b', marker='o', linestyle='None', markersize=2,markeredgecolor=None)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(f'{label} mole fraction')

ax = fig.add_subplot(gs[1, 3])
ax.plot(parameters[:,1],parameters[:,2] ,  color='b', marker='o', linestyle='None', markersize=2,markeredgecolor=None)
ax.set_xlabel(r'H$_2$ mole fraction')
ax.set_ylabel(r'CO$_2$ mole fraction')

plt.savefig(out_dir / f'sobol_parameter_space_{alpha}.png', dpi=300, bbox_inches='tight')

# purity of each point: the mole fraction of its most abundant species

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(parameters[:,cols].max(axis=1), bins=50, color='b', edgecolor='k')
ax.set_xlabel('Largest mole fraction in mixture')
ax.set_ylabel('Count')

plt.savefig(out_dir / 'sobol_purity.png', dpi=300, bbox_inches='tight')
plt.show()

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import csv
import optunity
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

num_pts = 2 ** 14   # the number of points to generate
skip = 500    # the number of initial points of the Sobol sequence to skip
N = 3 # number of dimensions

out_dir = REPO_ROOT / 'results' / 'sobol_sequence'
out_dir.mkdir(parents=True, exist_ok=True)

# paramter bounds

temp = [220, 300] # temperature bounds in celcius
H2_CO2_ratio = [0.5, 6] # ratio of H2 and CO2
perc_Ar = [0, 0.75] # percent argon

# generate Sobol sequence

x_sobol=np.zeros((num_pts, N))
res = optunity.solvers.Sobol.i4_sobol_generate(N, num_pts, skip)
x_sobol = np.array(res)

# apply the sobol sequence to the parameter bounds

parameters = x_sobol.copy()

parameters[:,0] = temp[0] + parameters[:,0] * (temp[1] - temp[0])
parameters[:,1] = H2_CO2_ratio[0] + parameters[:,1] * (H2_CO2_ratio[1] - H2_CO2_ratio[0])
parameters[:,2] = perc_Ar[0] + parameters[:,2] * (perc_Ar[1] - perc_Ar[0])

np.savetxt(out_dir / 'sobol_points.csv', parameters, delimiter=',', fmt='%f', header='Temperature,H2_CO2_ratio, %Ar', comments='')

colormap=plt.cm.Dark2
colors = [colormap(i) for i in np.linspace(0, 1, 4)]

gs=gridspec.GridSpec(nrows=1, ncols=3)
gs.update(wspace=0.35, hspace=0.15)

ax0=plt.subplot(gs[0,0])
ax1=plt.subplot(gs[0,1])
ax2=plt.subplot(gs[0,2])
ax0.plot(parameters[:,0],parameters[:,1] ,  color='b', marker='o', linestyle='None', markersize=2,markeredgecolor=None)
ax1.plot(parameters[:,0],parameters[:,2] ,  color='b', marker='o', linestyle='None', markersize=2,markeredgecolor=None)
ax2.plot(parameters[:,1],parameters[:,2] ,  color='b', marker='o', linestyle='None', markersize=2,markeredgecolor=None)
ax0.set_xlabel('Temperature (°C)')
ax0.set_ylabel(r'H$_2$/CO$_2$ ratio')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('%Ar')
ax2.set_xlabel(r'H$_2$/CO$_2$ ratio')
ax2.set_ylabel('%Ar')
# ax0.set_xlim([0,1])
# ax1.set_xlim([0,1])
# ax2.set_xlim([0,1])
# ax0.set_ylim([0,1])
# ax1.set_ylim([0,1])
# ax2.set_ylim([0,1])

# save the plot to results/sobol_sequence
plt.savefig(out_dir / 'sobol_parameter_space.png', dpi=300, bbox_inches='tight')
plt.show()

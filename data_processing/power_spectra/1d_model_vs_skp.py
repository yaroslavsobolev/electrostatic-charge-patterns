import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.legend import Legend

f1 = plt.figure(figsize=(5,8))

prefactor = 1
lower_cutoff = 0
upper_cutoff = 4e10
color = 'C0'
label = 'Model ($\sigma_{0}=7$ nC cm$^{-2}$)'
alpha=0.5
filename = '{0:03d}.npy'.format(0)
data = np.load('theoretical_model/'+filename)
xs = data[0, :]
ys = data[1, :]
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
# mask = np.logical_and(mask, ys > 0)
plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)

prefactor = 1
lower_cutoff = 0
upper_cutoff = 4e10
color = 'C1'
label = 'SKP (81% RH)'
alpha=1
filename = '{0:03d}.npy'.format(0)
data = np.load('kelvinprobe/1d/005_test.npy')
xs = data[0, :]
ys = data[1, :]
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
# mask = np.logical_and(mask, ys > 0)
plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)


color = 'C2'
label = 'SKP (78% RH)'
alpha=1
filename = '{0:03d}.npy'.format(0)
data = np.load('kelvinprobe/1d/20191123_70RH_B01.npy')
xs = data[0, :]
ys = data[1, :]
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
# mask = np.logical_and(mask, ys > 0)
plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)


color = 'C3'
label = 'SKP (68% RH)'
alpha=1
filename = '{0:03d}.npy'.format(0)
data = np.load('kelvinprobe/1d/20191015_62RH_B01.npy')
xs = data[0, :]
ys = data[1, :]
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
# mask = np.logical_and(mask, ys > 0)
plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)

plt.xlabel('Spatial frequency [mm$^{-1}$]')
plt.ylabel('Power spectral density, (nC cm$^{{-2}}$)$^2$ mm')
plt.legend()
plt.yscale('log')
plt.xscale('log')#, subsx=[-1, 0, 1, 2, 3, 4])
plt.xlim(2e-2, 5)
plt.ylim(1e-2, 4e2)
plt.grid(True,which="major",ls="-")
plt.tight_layout()

f1.savefig('figures/1d_psd_comparison.png', dpi=300)
plt.show()
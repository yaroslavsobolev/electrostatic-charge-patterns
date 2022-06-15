import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy.interpolate import PPoly
import pickle

f_pash, ax = plt.subplots(1, figsize=(11,7.5))
plt.xscale('log')
plt.yscale('log')

data_small_gaps = np.loadtxt('argon/Radmilovic-Radjenovic__small_gaps_at_995mbar_fromFig3a.txt', skiprows=2)
data_large_gaps = np.loadtxt('argon/meek_and_craggs__corrected_to_1atm.txt', skiprows=2)[9:]

gaps_in_mm = 1e-3*data_small_gaps[:, 0]
field_in_kV_per_mm = 1e-3*data_small_gaps[:,1]/(1e-3*data_small_gaps[:,0])
ax.scatter(gaps_in_mm, field_in_kV_per_mm, label='Small gaps [Radmilovic-Radjenovic et al., 2013]', marker='o', s=20, alpha=0.5)
all_data = np.vstack((gaps_in_mm[:-2], field_in_kV_per_mm[:-2])).T

gaps_in_mm = 1e-3*data_large_gaps[:, 0]
field_in_kV_per_mm = 1e-3*data_large_gaps[:,1]/(1e-3*data_large_gaps[:,0])
ax.scatter(gaps_in_mm, field_in_kV_per_mm, label='Large gaps [Meek & Craggs, 1953]', marker='o', s=20, alpha=0.5)

all_data = np.concatenate((all_data,
                          np.vstack((gaps_in_mm, field_in_kV_per_mm)).T),
                          axis=0)


ax.scatter(5e-5, 1e3, label='Vacuum limit', marker='o', s=20, alpha=0.5)
all_data = np.concatenate((all_data,
                          np.vstack((np.array([5e-5, 1e-5]), np.array([1e3, 1e3]))).T),
                          axis=0)

all_data = np.concatenate((all_data,
                          np.vstack((np.array([13.157, 131.57]), np.array([0.684, 0.684]))).T),
                          axis=0)

def make_spline_and_save():
    all_data[:,0] = all_data[:,0] + np.random.rand(all_data.shape[0])*1e-9
    data_sorted = all_data[all_data[:,0].argsort()]
    x = np.log10(data_sorted[:,0])
    y = np.log10(data_sorted[:,1])
    weights = (x+5)**1.3
    weights[0]=20
    weights[1]=20
    weights[2]=20
    weights[3]=30
    # weights[78]=50
    # weights[138]=50
    spl = UnivariateSpline(x, y, w=weights, s=1)
    spl2 = splrep(x, y, w=weights, s=1)
    pickle.dump(spl2, open("paschen_argon.pickle", "wb"))

    xs = np.logspace(-4.5, 0.477, 1000)
    ax.plot(xs, np.power(10, spl(np.log10(xs))), 'g', lw=8, alpha=0.3)
    ax.plot(xs, np.power(10, splev(np.log10(xs), spl2)), 'b', lw=12, alpha=0.3)

# make_spline_and_save()
xs = np.logspace(-4.5, 1, 1000)
spl3 = pickle.load(open("paschen_argon.pickle", "rb"))
ax.plot(xs, np.power(10, splev(np.log10(xs), spl3)), 'grey', lw=3, alpha=0.3, zorder=-8,
        label='Empirical interpolating function for argon')

# xs = np.logspace(-4.5, 1, 1000)
# spl4 = pickle.load(open("paschen_air.pickle", "rb"))
# ax.plot(xs, np.power(10, splev(np.log10(xs), spl4)), '--', 'grey', lw=2, alpha=0.2, zorder=-8,
#         label='Air')

plt.ylim([0.1, 5000])
plt.legend()
plt.ylabel('Breakdown field, kV/mm')
plt.xlabel('Gap between electrodes, mm')
f_pash.savefig('air_paschen_data_and_interpolator_for_argon.png', dpi=600)
plt.show()
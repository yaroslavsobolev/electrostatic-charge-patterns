import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

frac_dims_data = np.loadtxt('LBC_fractal_dimension.txt', skiprows=1, delimiter='\t')
epsilons = frac_dims_data[:,0]
frac_dims = frac_dims_data[:,1]
logeps = np.log10(epsilons)
get_frac_dim = interp1d(logeps, frac_dims, kind='linear')
def get_spectral_power_exponent(spatial_frequency): #spatial frequency in inverse meters
    spatial_scale = 1/spatial_frequency # in meters
    eps_here = spatial_scale/4.5e-6*1024
    try:
        frac_dim_here = get_frac_dim(np.log10(eps_here))
    except ValueError:
        print('Outside of interpolator. Eps = {0}'.format(eps_here))
        if eps_here < 10:
            frac_dim_here = 0.8173
        elif eps_here > 1000:
            frac_dim_here = 2.0015
        else:
            print('Weird value of epsilon')
            raise Exception

    power_exponent = 6-2*frac_dim_here
    return power_exponent

xs = np.logspace(np.log10(1/4.5e-3), np.log10(1/(4.5e-3/10000)), num=100, base=10)
prefactor = 12e8
x0 = xs[0]
y0 = prefactor/x0**get_spectral_power_exponent(x0/(1e-3))
ys = [y0]
for x in xs[1:]:
    #assuming y = prefactor/x**exponent, find prefactor
    exponent_here = get_spectral_power_exponent(x/1e-3)
    prefactor = y0*x0**exponent_here
    print('Exponent_here={0}, prefactor_here={1}'.format(exponent_here, prefactor))
    y = prefactor/(x**exponent_here)
    y0 = y
    x0 = x
    ys.append(y)

plt.loglog(xs, ys, 'o-')
plt.grid(True,which="major",ls="-")
plt.show()
#
# ys = 12e8*xs**(-2)
# print(get_spectral_power_exponent(1/(10e-9)))
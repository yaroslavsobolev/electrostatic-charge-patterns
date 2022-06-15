import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.legend import Legend

# f1, ax = plt.subplots(figsize=(7,7))
f1, ax = plt.subplots(figsize=(4.3,3.5))
plots = []
plt.yscale('log')
plt.xscale('log', subsx=[-1, 0, 1, 2, 3, 4])
plt.xlabel('Spatial frequency (mm$^{-1}$)')
plt.ylabel('Power spectral density (a.u.)')

#+
alpha = 0.5
prefactor = 800
lower_cutoff = 1e-1/1.41#0.31
upper_cutoff = 1
color = 'C0'
label = 'SKP'
for i in range(3):
    filename = '{0:03d}.npy'.format(i)
    data = np.load('kelvinprobe/'+filename)
    xs = data[0,:]
    ys = data[1,:]*(2*np.pi*xs)
    mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
    plots += plt.plot(xs[mask], prefactor*ys[mask], 'o', markersize=2, color=color, label=label, alpha=alpha)
    label = None

#+
prefactor = 4.7e-5*190
lower_cutoff = 0.4/1.41#0.67
upper_cutoff = 200 #45 #30
color = 'C1'
label = 'SEM'
local_prefactors = [1,1.5,3.5,1,1,1,1,0.5]
# local_prefactors = [1,1,1,1,1]
for i in [0,2,7]:
    filename = '{0:03d}.npy'.format(i+1)
    data = np.load('SEM/'+filename)
    xs = data[0, :]
    ys = data[1, :]*(2*np.pi*xs)
    if i == 2:
        mask = np.logical_and(xs > lower_cutoff, xs < 25)
    else:
        mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
    plots += plt.plot(xs[mask], local_prefactors[i]*prefactor*ys[mask],  color=color, label=label, alpha=alpha)
    label=None
#+
prefactor = 4e-12*2e-4
lower_cutoff = 4e2/1.41
upper_cutoff = 4e3/1.41
color = 'C2'
label = None
alpha=1
filename = '{0:03d}.npy'.format(1)
data = np.load('KPFM/'+filename)
xs = data[0, :]
ys = data[1, :]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
mask = np.logical_and(mask, ys > 0)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)
#+
prefactor = 1.3e-8*0.013
lower_cutoff = 70/1.41
upper_cutoff = 350/1.41
color = 'C2'
label = 'KPFM'
alpha=1
filename = '{0:03d}.npy'.format(2)
data = np.load('KPFM/'+filename)
xs = data[0, :]
ys = data[1, :]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
mask = np.logical_and(mask, ys > 0)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)

prefactor = 3e9
lower_cutoff = 0.35e-8
upper_cutoff = 4
color = 'C4'
label = 'Model ($\sigma_{0}=7$ nC cm$^{-2}$)'
alpha=1
filename = '{0:03d}.npy'.format(0)
data = np.load('theoretical_model/'+filename)
xs = data[0, :]
ys = data[1, :]
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
# mask = np.logical_and(mask, ys > 0)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)
#+
prefactor = 2.2e-15*4e-6
lower_cutoff = 2500
upper_cutoff = 22000
color = 'blue'
label = 'Siek $et$ $al.$, 2018'
alpha=1
filename = '{0:03d}.npy'.format(3)
data = np.load('KPFM/'+filename)
xs = data[0, :]
ys = data[1, :]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
mask = np.logical_and(mask, ys > 0)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)

# Shinbrot_Komatsu_Zhao
prefactor = 2e15
lower_cutoff = 1e-1
upper_cutoff = 5
color = 'C3'
alpha = 0.4
label = 'Shinbrot $et$ $al$., 2008'
filename = '{0:03d}.npy'.format(1)
data = np.loadtxt('Literature/Shinbrot_Komatsu_Zhao.csv', skiprows=2, delimiter=',')
xs = data[:, 0]/10
ys = data[:, 1]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
plots += plt.plot(xs[mask], prefactor*ys[mask], 'o', markersize=2, color=color, alpha=alpha)
xs = data[:, 2]/10
ys = data[:, 3]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
plots += plt.plot(xs[mask], prefactor*ys[mask], 'o', markersize=2, color=color, label=label, alpha=alpha)

# Baytekin et al. 2011
# #single exponent at >0.45 um
# xs = np.array([1/4.5e-3, 1/0.45e-3])
# ys = 12e8*xs**(-2)
# plots += plt.plot(xs, ys, '-', color='C6', alpha=0.5, linewidth=6, label='Baytekin $et$ $al.$, 2011', zorder=-100)
# continuous conversion from LBC fractal dimension to power exponent
prefactor = 12e8
lower_cutoff = 1/4.5e-3 # 0.67
upper_cutoff = 1/(4.5e-3/200)
color = 'C6'
alpha = 0.5
label = 'Baytekin $et$ $al.$, 2011'
frac_dims_data = np.loadtxt('Literature/Baytekin/LBC_fractal_dimension.txt', skiprows=1, delimiter='\t')
epsilons = frac_dims_data[:,0]
frac_dims = frac_dims_data[:,1]
logeps = np.log10(epsilons)
get_frac_dim = interp1d(logeps, frac_dims, kind='linear')
def get_spectral_power_exponent(spatial_frequency): #spatial frequency in inverse meters
    spatial_scale = 1/(spatial_frequency*2)/np.sqrt(2) # in meters
    eps_here = spatial_scale/4.5e-6*1000
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
xs = np.logspace(np.log10(lower_cutoff), np.log10(upper_cutoff), num=100, base=10)
x0 = xs[0]
y0 = prefactor/x0**get_spectral_power_exponent(x0/(1e-3))
ys = [y0]
for x in xs[1:]:
    # assuming y = prefactor/x**exponent, find prefactor
    exponent_here = get_spectral_power_exponent(x/1e-3)
    prefactor = y0*x0**exponent_here
    print('Exponent_here={0}, prefactor_here={1}'.format(exponent_here, prefactor))
    y = prefactor/(x**exponent_here)
    y0 = y
    x0 = x
    ys.append(y)
plots += plt.plot(xs, ys*(2*np.pi*xs), color=color, label=label, alpha=alpha, linewidth=6)

# Bertein, 1973
#+
prefactor = 4.7e-2*7e4
lower_cutoff = 0.04#1e-5#0.09/1.41#0.67
upper_cutoff = 0.25#0.25/1.41
color = 'C3'
alpha = 1
label = 'Bertein, 1973'
data = np.load('Literature/Bertein.npy')
xs = data[0, :]
ys = data[1, :]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)
#+
prefactor = 1.7e-2*3500
lower_cutoff = 0.13/1.41#0.67
upper_cutoff =1.15/1.41
color = 'C5'
alpha = 1
label = 'Hull, 1949'
data = np.load('Literature/Hull_1949.npy')
xs = data[0, :]
ys = data[1, :]*(2*np.pi*xs)
mask = np.logical_and(xs > lower_cutoff, xs < upper_cutoff)
plots += plt.plot(xs[mask], prefactor*ys[mask], color=color, label=label, alpha=alpha)

# fit_curve
xs = np.array([2e3, 5e4])
ys = 7e15*xs**(-4)*(2*np.pi*xs)
plots += plt.plot(xs, ys, '-', linestyle='dotted', color='black', alpha=0.7, linewidth=1)#, label='P~k$^{-4}$')

# fit_curve
xs = np.array([3e-2, 5e4])
ys = 12e8*xs**(-2)*(2*np.pi*xs)
plots += plt.plot(xs, ys, '-', color='black', alpha=1, linewidth=1)#, label='P~k$^{-2}$')

# # fit_curve
# xs = np.array([0.7e-1, 1])
# ys = 15e8*xs**(-1)*(2*np.pi*xs)
# plots += plt.plot(xs, ys, '--', color='gray', alpha=0.9, linewidth=1)#, label='P~k$^{-1}$')

f1.set_size_inches(7, 7, forward=True)
plt.grid(True,which="major",ls="-")
plt.xlim(1e-1/3, 1e1)
# plt.ylim(1e7, 2e11)
plt.tight_layout()
# f1.savefig('figures/combined_power_spectra_zoom_2b.png', dpi=200)
# plt.show()
plt.legend()

handles, labels = ax.get_legend_handles_labels()
print(len(handles))
split_legend_at_index = 4
leg1 = plt.legend(handles[:split_legend_at_index], labels[:split_legend_at_index],
          loc='upper right', title='Present work:', title_fontsize=11)
leg2 = plt.legend(handles[split_legend_at_index:], labels[split_legend_at_index:],
             loc='lower left', title='Literature data:', title_fontsize=11, framealpha=0)
leg1.get_frame().set_facecolor('white')
leg1._legend_box.align = "left"
# leg2.get_frame().set_facecolor('white')
leg2._legend_box.align = "left"
plt.gca().add_artist(leg1)

plt.xlim(1e-1/3, 2e4)
plt.ylim(1e4, 3e11)
plt.grid(True,which="major",ls="-")
plt.tight_layout()
f1.savefig('figures/combined_power_spectra_4b.png', dpi=300)
plt.show()

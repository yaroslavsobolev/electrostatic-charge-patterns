import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

list_of_filenames = [
'experimental_data\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
'experimental_data\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
'experimental_data\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p6_E01\\kelvinprobe',
'experimental_data\\improved_kp_kelvinprobe\\20191107_5cm_3in_41RH_ambRH36_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_B01',
'experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',
'experimental_data\\improved_kp_kelvinprobe\\20191030_5cm_3in_50RH_ambRH38_eq30min_newPDMS5to1_PMMAtol_uniformspeeds_0p5_C01_copy'
]

corrected_humidities = [80.94354838709671,
                        78.3897849462365,
                        80.94354838709671,
                        41.360215053763426,
                        68.17473118279565,
                        52.85215053763438]
sample_names = ['A', 'B', 'C', 'D', 'E', 'F']
fig, ax3 = plt.subplots(figsize=(7,6))
exp_lines = []
for i, fname in enumerate(list_of_filenames):
    data = np.load(fname + '/processed/moving_averaged_net_charge.npy')
    xs = data[0, :]
    ys = data[1, :]
    humidity = corrected_humidities[i]
    handle_here, = plt.loglog(xs, ys, alpha=0.5, linewidth=3,
                                label='Sample {0}, {1:.0f}% RH'.format(sample_names[i],
                                                                      humidity))
    exp_lines.append(handle_here)

second_legend_plots = []

def fit_power_law(x,y,yerr,ax3):
    def func_pl(x, a, b):
        return a*(x**b)
    popt_pl, pcov = curve_fit(func_pl, x, y, p0=[1e-2, 1.5],
                              sigma=yerr, absolute_sigma=True)
    ax3.plot(x, func_pl(x, popt_pl[0], popt_pl[1]), '--', color='black', alpha=0.5)
    perr = np.sqrt(np.diag(pcov))
    print('Power exponent is: {0:.2f} ± {1:.2f}'.format(popt_pl[1], perr[1]))
    return '$\\beta=${0:.2f} ± {1:.2f}'.format(popt_pl[1], perr[1])

#add data from [Apodaca et. al. 2010]
data = np.loadtxt('literature/Apodaca2010_PDMS_PDMS.txt',
                       skiprows=1, delimiter='\t')
data_plus_sigma = np.loadtxt('literature/Apodaca2010_PDMS_PDMS_plus_sigma.txt',
                       skiprows=1, delimiter='\t')
PDMS_PDMS_xs = data[:, 0]
PDMS_PDMS_ys = data[:, 1]/2
PDMS_PDMS_yerr = (data_plus_sigma[:, 1] - data[:, 1])/2
fit_string = fit_power_law(PDMS_PDMS_xs, PDMS_PDMS_ys, PDMS_PDMS_yerr, ax3)
handle_here = plt.errorbar(x=PDMS_PDMS_xs, y=PDMS_PDMS_ys, yerr=PDMS_PDMS_yerr, color='grey', alpha=1, zorder=-30,
             capsize=3, linestyle='none', marker='o',
            label='PDMS-PDMS, {0}'.format(fit_string))
second_legend_plots.append(handle_here)

# fn,axn = plt.subplots()
# PDMS_PDMS_xs = data[:, 0]
# PDMS_PDMS_ys = data[:, 1]/2
# print(np.polyfit(np.log10(PDMS_PDMS_xs**2), np.log10(PDMS_PDMS_ys), deg=1))
# plt.show()

data = np.loadtxt('literature/Apodaca2010_PDMS_PVC.txt',
                       skiprows=1, delimiter='\t')
sigmas = np.loadtxt('literature/Apodaca2010_PDMS_PVC_sigmas.txt',
                       skiprows=0, delimiter='\t')
PDMS_PVC_xs = data[:, 0]
PDMS_PVC_ys = data[:, 1]/2
PDMS_PVC_yerr = sigmas/2/2
fit_string = fit_power_law(PDMS_PVC_xs, PDMS_PVC_ys, PDMS_PVC_yerr, ax3)
handle_here = plt.errorbar(x=PDMS_PVC_xs, y=PDMS_PVC_ys, yerr=PDMS_PVC_yerr, color='grey', alpha=1, zorder=-30,
             capsize=3, linestyle='none', marker='^',
            label='PDMS-PVC, {0}'.format(fit_string))
second_legend_plots.append(handle_here)

data = np.loadtxt('literature/Apodaca2010_stainless.txt',
                       skiprows=1, delimiter='\t')
sigmas = np.loadtxt('literature/Apodaca2010_stainless_sigmas.txt',
                       skiprows=0, delimiter='\t')
PDMS_SS_xs = data[:, 0]
PDMS_SS_ys = data[:, 1]/2
PDMS_SS_yerr = sigmas/2
fit_string = fit_power_law(PDMS_SS_xs, PDMS_SS_ys, PDMS_SS_yerr, ax3)
handle_here = plt.errorbar(x=PDMS_SS_xs, y=PDMS_SS_ys, yerr=PDMS_SS_yerr, color='grey', alpha=1, zorder=-30,
             capsize=3, linestyle='none', marker='s',
            label='PDMS-steel, {0}'.format(fit_string))
second_legend_plots.append(handle_here)


ax3.plot([1, np.sqrt(10)], [0.1, np.sqrt(1)], color='black')
ax3.plot([np.sqrt(10), 10], [1e-2, np.sqrt(1e-3)], color='black')
# PDMS_PDMS_xs = PDMS_PDMS_xs[2:]
# PDMS_PDMS_ys = PDMS_PDMS_ys[2:]
# PDMS_PDMS_yerr = PDMS_PDMS_yerr[2:]



# cut_from = 8
# cut_to = 20
# xs_for_fit = xs[cut_from:cut_to]
# ys_for_fit = ys[cut_from:cut_to]
#
# def func_lin(x, a):
#     return a * x
#
# popt_lin, pcov = curve_fit(func_lin, xs_for_fit, ys_for_fit)
# ax3.plot(xs, func_lin(xs, popt_lin[0]), '--', color='C0')
#
# cut_from = 0
# cut_to = 8
# xs_for_fit = xs[cut_from:cut_to]
# ys_for_fit = ys[cut_from:cut_to]
#
# def func_sq(x, a):
#     return a * x ** 2
#
# popt_sq, pcov = curve_fit(func_sq, xs_for_fit, ys_for_fit)
# ax3.plot(xs, func_sq(xs, popt_sq[0]), '--', color='C1')

ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylim(3e-4, 4)
ax3.set_xlim(1e-1, 12)
plt.grid(True, which="major", ls="--")
ax3.set_ylabel(
    'Mean absolute value of window-integrated charge $\\langle A\cdot | \\langle \sigma \\rangle _{W}| \\rangle $, nC')
ax3.set_xlabel('Window dimension ($\sqrt{A}$), mm')

legend1 = plt.legend(handles=exp_lines, loc='upper left', title='PDMS-PMMA (our data):')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=second_legend_plots, loc='lower right', title='Charge of one surface\n[Apodaca et. al. 2010]:')
ax = plt.gca().add_artist(legend2)
plt.tight_layout()
fig.savefig('combined_moving_averages.png', dpi=300)
plt.show()
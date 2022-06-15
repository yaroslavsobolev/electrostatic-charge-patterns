import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy.interpolate import PPoly
import pickle

f_pash, ax = plt.subplots(1, figsize=(11,7.5))
plt.xscale('log')
plt.yscale('log')

def load_datafile_and_plot(filename, format_type, label, ax, marker='+', color=False):
    data = np.loadtxt(filename, skiprows=2)
    if format_type == 'format_A':
        gaps_in_mm = data[:, 0]
        field_in_kV_per_mm = data[:, 2]
    elif format_type == 'format_B':
        gaps_in_mm = 1e-3*data[:, 0]
        field_in_kV_per_mm = 1e-3*data[:,1]/(1e-3*data[:,0])
    marker_size_factor = 0.5
    if marker=='x':
        msize = 20*marker_size_factor
    else:
        msize = 40*marker_size_factor
    if not color:
        ax.scatter(gaps_in_mm, field_in_kV_per_mm, label=label, marker=marker, s=msize, alpha=0.5)
    else:
        ax.scatter(gaps_in_mm, field_in_kV_per_mm, label=label, marker=marker, s=msize, alpha=0.5, c=color)
    result = np.vstack((gaps_in_mm, field_in_kV_per_mm)).T
    return result

res = load_datafile_and_plot('literature/Bertein.txt', 'format_A', 'Bertein', ax)
all_data = np.copy(res)

# Schreier, Stefan. "On the Breakdown Voltages of Some Electronegative Gases at Low Pressures."
# IEEE Transactions on Power Apparatus and Systems 83.5 (1964): 468-471.
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/schreier.txt', 'format_A', 'Schreier', ax)),
                          axis=0)
# Hourdakis, E., Bryant, G. W., & Zimmerman, N. M. (2006). Electrical breakdown in the microscale:
# Testing the standard theory. Journal of Applied Physics, 100(12), 1–6. https://doi.org/10.1063/1.2400103
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/hourdakis_2006.txt', 'format_B',
                                                  'Hourdakis $et. al.$', ax, marker='x')),
                          axis=0)
#Seeliger, R. (1934). Einführung in die Physik der Gasentladungen. Leipzig: Johann Ambrosius Barth.
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/seeliger.txt', 'format_A',
                                                  'Seeliger', ax)),
                          axis=0)
#Hirata, Y., Ozaki, K., Ikeda, U., & Mizoshiri, M. (2007). Field emission current and vacuum breakdown by a pointed cathode.
# Thin Solid Films, 515(9), 4247–4250. https://doi.org/10.1016/j.tsf.2006.02.085
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/hirata_vacuum.txt', 'format_B', 'Hirata $et. al.$, vacuum',
                                                  ax, marker='x')),
                          axis=0)

#Peschot, A., Poulain, C., Bonifaci, N., & Lesaint, O. (2015). Electrical breakdown voltage in micro- and submicrometer
# contact gaps (100nm - 10μm) in air and nitrogen.
# Electrical Contacts, Proceedings of the Annual Holm Conference on Electrical Contacts,
# 2015-Decem, 280–286. https://doi.org/10.1109/HOLM.2015.7355110
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/peschot_2015_Au_air.txt', 'format_B',
                                                  'Peschot $et. al.$, Au electrode', ax, marker='x')),
                          axis=0)
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/peschot_2015_Ru_air.txt', 'format_B',
                                                  'Peschot $et. al.$, Ru electrode', ax, marker='x')),
                          axis=0)
# Lee, R.-T., Chung, H.-H., & Chiou, Y.-C. (2001). Arc erosion behaviour of silver electric contacts.
# IEE Proc. -Sci. Meas. Technol., 148(1), 8–14. https://doi.org/10.1049/ip-smt:20010181
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/lee_et_al_air_2001.txt', 'format_B',
                                                  'Lee $et. al.$, Ag electrode', ax, marker='x')),
                          axis=0)

#Torres, J. M., & Dhariwal, R. S. (1999). Electric field breakdown at micrometre separations.
# Nanotechnology, 10(1), 102–107. https://doi.org/10.1088/0957-4484/10/1/020
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/torres_et_al_air_aluminum.txt', 'format_B',
                                                  'Torres $et. al.$, Al electrode', ax, marker='x')),
                          axis=0)
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/torres_et_al_air_brass.txt', 'format_B',
                                                  'Torres $et. al.$, Brass electrode', ax, marker='x')),
                          axis=0)
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/torres_et_al_air_nickel.txt', 'format_B',
                                                  'Torres $et. al.$, Ni electrode', ax, marker='x')),
                          axis=0)

# J.D. Smith data in [Germer, L. H. (1959). Electrical breakdown between close electrodes in air.
# Journal of Applied Physics, 30(1), 46–51. https://doi.org/10.1063/1.1734973]
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/jd_smith_in_germer_1959.txt', 'format_B',
                                                  'J.D. Smith data in [Germer, 1959], Pd electrode', ax, marker='x')),
                          axis=0)

# Germer, L. H. (1959). Electrical breakdown between close electrodes in air.
# Journal of Applied Physics, 30(1), 46–51. https://doi.org/10.1063/1.1734973
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/germer_1959.txt', 'format_B',
                                                  'Germer, Pd electrode', ax, marker='x', color='C3')),
                          axis=0)

# Data from various standards
all_data = np.concatenate((all_data,
                           load_datafile_and_plot('literature/standards.txt', 'format_A',
                                                  'Publication IEC №52 (1960)\n'
                                                  'Norme italienne 42.1 (1963) (Italian Standard)\n'
                                                  'Norme C 41050 UTE (1960) (French Standard)\n'
                                                  'IEEE Standard №4 (ANSI. C 68.1)\n'
                                                  'British Standard 358 (1960)\n'
                                                  'VDE 0433 (German Standard)', ax, marker='1')),
                          axis=0)

# Make spline interpolator. Since it's iterative, it does not give the same result every time. Rerun it few times
# to get the spline that we eventually used here
def make_spline_and_save():
    all_data[:,0] = all_data[:,0] + np.random.rand(all_data.shape[0])*1e-9
    data_sorted = all_data[all_data[:,0].argsort()]
    x = np.log10(data_sorted[:,0])
    y = np.log10(data_sorted[:,1])
    weights = (x+5)**1.3
    weights[0]=20
    weights[7]=50
    weights[78]=50
    weights[138]=50
    spl = UnivariateSpline(x, y, w=weights, s=95)
    spl2 = splrep(x, y, w=weights, s=95)
    # pickle.dump(spl2, open("paschen_air.pickle", "wb"))

    xs = np.logspace(-4.5, 3, 1000)
    ax.plot(xs, np.power(10, spl(np.log10(xs))), 'g', lw=8, alpha=0.3)
    ax.plot(xs, np.power(10, splev(np.log10(xs), spl2)), 'b', lw=12, alpha=0.3)

xs = np.logspace(-4.5, 3, 1000)
spl3 = pickle.load(open("paschen_air.pickle", "rb"))
ax.plot(xs, np.power(10, splev(np.log10(xs), spl3)), 'grey', lw=3, alpha=0.3, zorder=-8,
        label='Empirical interpolating function')

plt.ylabel('Breakdown field, kV/mm')
plt.xlabel('Gap between electrodes, mm')
plt.ylim([1, 5000])
plt.legend()
f_pash.savefig('air_paschen_data_and_interpolator.png', dpi=600)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
import matplotlib.ticker as plticker
import matplotlib.cm as cm
import glob
import os
import re
from scipy import integrate
from hygrometer_calibration.calibration_function import correct_humidity

gravity = 9.79776 #m/s^2 -- this is local gravity in Ulsan, Republic of Korea

def get_net_charge(target_folder, do_plotting=True, deriv_thresh = 6e-4, shift_value = 300, bkg_from='start', plotname='default.png'):
    try:
        file_name = glob.glob(target_folder + "electrometer/*values_only.txt")[0]
    except IndexError:
        print('ELECTROMETER FILE NOT FOUND IN {0}'.format(target_folder))
        return np.nan
    data = np.loadtxt(file_name)
    rps = np.loadtxt(target_folder + 'electrometer/readings_per_second.txt')
    Npoints = data.shape[0]
    t = np.linspace(0, Npoints/rps, Npoints)
    deriv = signal.savgol_filter(data, window_length=101, polyorder=2, deriv=1)
    # start_index = np.argmax(deriv > deriv_thresh)
    # end_index = deriv.shape[0] - np.argmax(np.flip(deriv) > deriv_thresh)
    # print(start_index)
    # print(end_index)
    # line_fit_params = np.polyfit(t[:start_index - shift_value], data[:start_index - shift_value], 1)
    # line_fit_function = np.poly1d(line_fit_params)
    # xs = np.linspace(t[start_index-shift_value], t[end_index+shift_value], 100)
    # ys = line_fit_function(xs)
    # net_charge = data[end_index+shift_value] - line_fit_function(t[end_index+shift_value])
    end_of_delamination = np.argmax(np.abs(deriv))
    if bkg_from == 'start':
        line_fit_params = np.polyfit(t[end_of_delamination - int(round(rps*50)):end_of_delamination - int(round(rps*30))],
                                     data[end_of_delamination - int(round(rps*50)):end_of_delamination - int(round(rps*30))],
                                     1)
        line_fit_function = np.poly1d(line_fit_params)
        xs = np.linspace(t[0], t[-1], 100)
        ys = line_fit_function(xs)
        where_to_evaluate = end_of_delamination + int(round(rps*10))
        if where_to_evaluate > data.shape[0] - 2:
            where_to_evaluate = data.shape[0] - 2
        net_charge = data[where_to_evaluate] - line_fit_function(t[where_to_evaluate])
    elif bkg_from == 'end':
        line_fit_params = np.polyfit(t[end_of_delamination + int(round(rps*15)):end_of_delamination + int(round(rps*25))],
                                     data[end_of_delamination + int(round(rps*15)):end_of_delamination + int(round(rps*25))],
                                     1)
        where_to_evaluate = end_of_delamination - int(round(rps*30))
        if where_to_evaluate < 1:
            where_to_evaluate = 1
        line_fit_function = np.poly1d(line_fit_params)
        xs = np.linspace(t[0], t[-1], 100)
        ys = line_fit_function(xs)
        net_charge = line_fit_function(t[where_to_evaluate]) - data[where_to_evaluate]
    if do_plotting:
        fig, axarr = plt.subplots(2,1, sharex=True)
        axarr[1].plot(t, deriv)#*rps)
        # axarr[1].axvline(x=t[start_index], color='black')
        axarr[1].axvline(x=t[end_of_delamination], color='black')
        axarr[1].set_xlabel('Time, s')
        axarr[1].set_ylabel('Current, nA')
        axarr[0].plot(t,  data)
        axarr[0].plot(xs, ys)
        axarr[0].set_ylabel('Charge, nC')
        pieces = target_folder.split('/')
        lens = [len(x) for x in pieces]
        longest_piece = pieces[np.argmax(np.array(lens))]
        axarr[0].set_title('{0}\nCharge: {1:.5f} nC\nEvaluate from: {2}'.format(longest_piece, net_charge,
                                                           bkg_from), fontsize=7) #.replace('_', '\n')
        fig.savefig('electrometer_figures/net_charge/{0}.png'.format(plotname))
        # plt.show()
        plt.close(fig)

        electrometer_to_balance_delay = np.loadtxt(target_folder + 'electrometer/electrometer_to_balance_delay.txt')
        force_data = np.loadtxt(target_folder + "force/force_vs_time.txt", usecols=[0,1,2], skiprows=2)
        t0 = force_data[np.argmax(force_data[:,2]),0] - electrometer_to_balance_delay
        fig, ax = plt.subplots(figsize=(8, 3))
        color='black'
        ax.plot(force_data[:,0] - electrometer_to_balance_delay-t0,
                gravity*(0.001)*force_data[:,2], color=color)
        np.save(target_folder + 'force/force_and_coulombmeter_vs_time_t',
                force_data[:,0] - electrometer_to_balance_delay-t0)
        np.save(target_folder + 'force/force_and_coulombmeter_vs_time_f',
                gravity*(0.001)*force_data[:,2])
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('Pull force, N')
        ax2 = ax.twinx()
        color = 'C0'
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Coulomb-meter reading, nC', color=color)
        ax2.plot(t-t0, data, color=color)
        ax2.plot(xs-t0, ys, '--', color=color, alpha=0.4)
        ax2.spines['right'].set_color(color)
        ax2.set_xlim(-3,
                     t[end_of_delamination]+2-t0)
        ax2.set_ylim(-1,3.1)
        ax.axvspan(0,
                     t[end_of_delamination]-t0, color='yellow', alpha=0.2)
        ax.set_xlabel('Time, s')

        plt.tight_layout()
        fig.savefig(target_folder + 'force/force_and_coulombmeter_vs_time.png', dpi=300)
        # plt.show()
        plt.close(fig)

    return net_charge

def generate_lines(filePath, delimiters=[]):
    with open(filePath) as f:
        for line in f:
            line = line.strip()  # removes newline character from end
            for d in delimiters:
                line = line.replace(d, " ")
            yield line

def get_adhesion_work(target_folder, do_plotting=True, plotname='default.png'):
    try:
        file_name = glob.glob(target_folder + "force/force_vs_time.txt")[0]
    except IndexError:
        print('FORCE FILE NOT FOUND IN {0}'.format(target_folder))
        return np.nan
    gen = generate_lines(file_name, delimiters=['\t', ' '])
    data = np.loadtxt(gen, usecols=[0,1,2], skiprows=2) # time (s), position (mm), force (grams)

    # Find where delamination begins -- we assume that it is the point of maximum force
    delamination_start_index = np.argmax(data[:,2])

    # find where delamination ends. We assume that it is the first positive force from the end of trace
    delamination_end_index = data.shape[0] - 1 - np.argmax(np.flip(data[:,2])>0)
    delamination_time = data[delamination_end_index, 0] - data[delamination_start_index, 0]

    # integrate with simps
    force = gravity*(0.001)*data[delamination_start_index:delamination_end_index + 1, 2] # in Newtons
    coordinate = -0.001*data[delamination_start_index:delamination_end_index + 1, 1] # in meters
    work = integrate.trapz(y=force, x=coordinate)
    area = 5e-4 # 5 cm^2 in m^2
    work_per_area = work/area
    print('Work per area: {0:.2f} mJ/m^2, time={1:.2f}'.format(work_per_area/1e-3, delamination_time))
    fig, ax = plt.subplots(figsize=(6,3))
    plt.plot(-1*data[:,1], gravity*(0.001)*data[:,2], color='black')
    plt.scatter(-1 * data[:, 1], gravity * (0.001) * data[:, 2], s=12, color='black')
    # plt.plot(-1*data[delamination_start_index:delamination_end_index + 1, 1],
    #             data[delamination_start_index:delamination_end_index + 1, 2], color='yellow')
    plt.fill_between(x=-1*data[delamination_start_index:delamination_end_index + 1, 1],
                y1=0,
                y2=gravity*(0.001)*data[delamination_start_index:delamination_end_index + 1, 2], color='yellow',
                                alpha=0.5)
    plt.axhline(y=0, color='grey')
    plt.axvline(x=-1*data[delamination_start_index, 1], color='C2')
    plt.axvline(x=-1*data[delamination_end_index, 1], color='C3')
    plt.ylabel('Pull force, N')
    plt.xlabel('Vertical position of the stage, mm')
    plt.xlim(np.min(-1*data[:,1]), -1*data[delamination_end_index + 1, 1]+1)
    plt.tight_layout()
    fig.savefig('force_curve_figures/per_experiment/{0}.png'.format(plotname))
    fig.savefig(target_folder + 'force/adhesion_work_plot.png', dpi=300)
    # plt.show()
    plt.close(fig)
    return work_per_area, delamination_time


test_folder = 'Y:\\PDMS-PMMA_delamination_experiments\\data\\' \
              '20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01\\'
# test_folder = 'Y:\\PDMS-PMMA_delamination_experiments\\data\\' \
#               '20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p6_E01\\'
#
# get_adhesion_work(test_folder)
get_net_charge(target_folder=test_folder)

def process_all_experiments():
    directories = glob.glob("Y:/PDMS-PMMA_delamination_experiments/data/*3in*newPDMS*/")
    directories = [x.replace('\\', '/') for x in directories if ('Argo' not in x) and ('forAFM' not in x)]
    bkg_from_list = ['start']*125
    for i in [16, 17, 18, 22, 23, 32, 37, 38, 39, 43, 46, 47, 48, 49, 51, 52, 54, 56, 60, 64, 66, 67, 69, 71, 72, 74,
              79, 84, 93, 94, 98, 101, 107, 108, 109, 110, 111, 113, 114, 115, 117, 118]:
        bkg_from_list[i-4] = 'end'

    experiment_data = []
    for n, target_dir in enumerate(directories):
        print('{0} >>> {1}'.format(n, target_dir))
        # try:
        net_charge = get_net_charge(target_dir, bkg_from=bkg_from_list[n], plotname='{0}'.format(n))
        if np.isnan(net_charge):
            continue
        pieces = target_dir.split('/')
        lens = [len(x) for x in pieces]
        longest_piece = pieces[np.argmax(np.array(lens))]
        groups = re.findall(r'''in_(\d+)RH''', longest_piece)
        humidity = int(groups[0])
        groups = re.findall(r'''0p(\d)''', longest_piece)
        speed = 0.1*int(groups[0])

        work_per_area, delamination_time = get_adhesion_work(target_dir, plotname='{0:03d}'.format(n))
        sample_length = 5e-2 # 5 cm in meters
        horizontal_speed = sample_length/delamination_time # in m/s
        experiment_data.append([humidity, speed, net_charge, work_per_area, horizontal_speed])

        # except IndexError:
        # print('IndexError')
        # continue
    print(directories)
    # pickle.dump(favorite_color, open("save.p", "wb"))
    np.savetxt('extracted_experiment_data.txt', np.array(experiment_data), delimiter='\t')

# process_all_experiments()

fig, axarr = plt.subplots(2,1, sharex=True, figsize=(3.3, 6))

ax = axarr[0]
hist_params_vs_humidity = np.load('histogram_metrics_vs_humidity.npy')
rh_true = hist_params_vs_humidity[:,0]
ax.plot(rh_true, hist_params_vs_humidity[:,1], 'o', label='(SKP) Positive peak on histogram of charge density', color='C1')
ax.plot(rh_true, hist_params_vs_humidity[:,2], 'o', label='(SKP) Maximum value of positive charge density', color='C2')
the_ticks = ax.get_xticks()
# the_labels = ax.get_xlabels()

ax = axarr[0]
data = np.loadtxt('extracted_experiment_data.txt', delimiter='\t')
data = np.array([d for d in data if (not np.isclose(d[0], 29))])

rh_true = [correct_humidity(x, calibration_folder='hygrometer_calibration/') for x in data[:,0]]
data[:, 0] = rh_true
area = 5 # in cm^2

# for speed in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#     for_plotting = np.array([d for d in data if np.isclose(d[1], speed)])
#     ax.scatter(for_plotting[:, 0], for_plotting[:, 2]/area, s=10, label=speed, color='grey', alpha=0.6)
# # locc = ax.xaxis.get_major_locator()
medianprops = dict(linestyle='-.', linewidth=2.5, color='C0')
binsize = 5
for hum in list(range(5, 100, 5)):
    for_plotting = np.array([d for d in data if ((d[0] < hum + 0.5*binsize) and (d[0] > hum - 0.5*binsize))])
    if for_plotting.size > 0:
        ax.boxplot(for_plotting[:,2]/area, positions=[hum], widths=3,
                   medianprops=medianprops)

ax.set_ylabel('Net charge per area, nC/cm$^2$')
ax.axvspan(35, 85, alpha=1, color='lemonchiffon')
ax.plot([10, 70], [1.7, 6], '--', color='grey', alpha=0.8)
plt.legend()

ax = axarr[1]
for hum in list(range(5, 100, 5)):
    for_plotting = np.array([d for d in data if ((d[0] < hum + 0.5*binsize) and (d[0] > hum - 0.5*binsize) and (d[3] > 1e-3))])
    if for_plotting.size > 0:
        # ax.scatter(for_plotting[:, 0], for_plotting[:, 3], s=10, color='grey', alpha=0.6)
        ax.boxplot(for_plotting[:,3], positions=[hum], widths=3,
                   medianprops=medianprops)
ax.set_ylabel('Adhesion work per area, J/m$^2$')

for ax in axarr:
    tickpositions = range(0, 110, 10)
    ax.set_xticks(tickpositions)
    ax.set_xticklabels(['{0}'.format(x) for x in tickpositions])

ax.set_xlabel('Relative humidity, %')
plt.tight_layout()
# for ax in axarr:
#     ax.set_xticks(the_ticks)
figsize_0 = (6, 3.5)
fig_allspeeds,ax = plt.subplots(figsize=figsize_0)
colors = cm.viridis(np.linspace(0, 0.9, 6))
for i,speed in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    for_plotting = np.array([d for d in data if ( (d[3] > 1e-3) and (np.isclose(d[1], speed)) )])
    ax.scatter(for_plotting[:, 0], for_plotting[:, 2] / area, s=10, label='{0} mm/s'.format(speed),
               alpha=1, color=colors[i])
    # ax.scatter(for_plotting[:, 0], for_plotting[:, 3], s=10, label='V={0}'.format(speed), alpha=0.6)
plt.xlim(0, 100)
ax.set_xlabel('Relative humidity, %')
ax.set_ylabel('Net charge per area, nC/cm$^2$')
plt.legend()
plt.tight_layout()
fig_allspeeds.savefig('electrometer_figures/net_charge_vs_humidity_and_speeds_3.png', dpi=300)
plt.show()


# PLOTTING WORK FUNCTION VS RH FOR A SINGLE PULLING SPEED
fig0,ax = plt.subplots(figsize=figsize_0)
speed = 0.4
for_plotting = np.array([d for d in data if ( (d[3] > 1e-3) and (np.isclose(d[1], speed)) )])
# axarr[1].scatter(for_plotting[:, 0], for_plotting[:, 3], s=10, label=speed, alpha=0.6)
for hum in [correct_humidity(x, calibration_folder='hygrometer_calibration/') for x in [58, 60, 50, 39, 82, 70, 67, 18, 6]]:
    data_here = [d for d in for_plotting if np.isclose(d[0], hum)]
    data_here = np.array(data_here)
    mean = np.mean(data_here[:,3])
    stdev = np.std(data_here[:,3])
    ax.errorbar(x=hum, y=mean, yerr=stdev, capsize=5, elinewidth=2, markeredgewidth=2, color='C0')
    ax.scatter(x=hum, y=mean, s=50, color='C0')
ax.set_ylabel('Adhesion work per area, J/m$^2$')
ax.set_xlabel('Relative humidity, %')
ax.set_xlim(0, 100)
plt.tight_layout()

fig0.savefig('electrometer_figures/net_charge_work_humidity_boxplots_3.png', dpi=300)
plt.show()

fig2, ax = plt.subplots(figsize=figsize_0)
ax.scatter(data[:,1], data[:,4]/0.001, s=10, alpha=0.5)
ax.set_xlabel('Velocity of pull (vertical), mm/s')
ax.set_ylabel('Average front velocity (horizontal), mm/s')
plt.tight_layout()
fig2.savefig('electrometer_figures/vertical_speed_vs_front_velocity_2.png', dpi=300)
# plt.show()


fig3, ax = plt.subplots(figsize=figsize_0)
ax.scatter(data[:,4]/0.001, data[:,3], s=10, alpha=0.5)
ax.set_ylabel('Adhesion work per area, J/m$^2$')
ax.set_xlabel('Average front velocity (horizontal), mm/s')
plt.tight_layout()
fig3.savefig('electrometer_figures/adhesion_work_vs_front_velocity_2.png', dpi=300)
plt.show()

# net_charge = get_net_charge('E:/PDMS-PMMA_delamination_experiments/data/' \
#                 '20191017_5cm_3in_62RH_eq30min_oldUVPDMS_PMMAtol_uniformspeeds_0p1_B01/')
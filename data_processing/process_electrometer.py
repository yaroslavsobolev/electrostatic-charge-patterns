import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.ticker as plticker
import glob
# target_folder = 'D:/Docs/Science/UNIST/Projects/Vitektrification/combined_observation_experiments/20191009_5cm_3in_60RH_eq30min_dryPDMS_PMMAtol_uniformspeeds_0p075_B01/'
# target_folder = 'experimental_data/PDMS-PMMA_delamination_experiments/data/20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01/'
target_folder = 'E:/PDMS-PMMA_delamination_experiments/20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_' \
                'newwire_uniformspeeds_slowstart_dark_0p4_A01/'

data = np.loadtxt(glob.glob(target_folder + "electrometer/*values_only.txt")[0])
rps = np.loadtxt(target_folder + 'electrometer/readings_per_second.txt')
approx_noise = 0.01
Npoints = data.shape[0]
t = np.linspace(0, Npoints/rps, Npoints)
drift_indices = [0, 10000]
p = np.polyfit(t[drift_indices[0]:drift_indices[1]], data[drift_indices[0]:drift_indices[1]], deg=1)
drift = np.poly1d(p)
data = data - drift(t)
delamination_start_index = np.argmax(signal.savgol_filter(data, 501, polyorder=2) > approx_noise)
delamination_start_index = int(round(49.5*rps))
print(t[delamination_start_index])

deriv1 = np.diff(data)
deriv2 = signal.savgol_filter(deriv1, 61, polyorder=2)
delamination_end_index = np.argmax(deriv2)
delamination_end_index = int(round(58.57*rps))
delamination_end = t[delamination_end_index]
print(delamination_end)
plt.show()
plt.subplot(211)
plt.plot(t, data)
plt.subplot(212)
plt.psd(data, 512, rps)
sig = data
# The FFT of the signal
sig_fft = fftpack.fft(sig)
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(sig.size, d=1/rps)
# INDUSTRIAL BUTTER FILTER
sos1 = signal.butter(10, (58, 62), 'bs', fs=rps, output='sos')
filtered_sig = signal.sosfilt(sos1, sig)
plt.psd(filtered_sig, 512, rps, color='red')
plt.figure(figsize=(6, 5))
plt.plot(t, sig, label='Original signal')
plt.plot(t, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')
filtered = filtered_sig
max_index = delamination_end_index-230
fnn, ax1a = plt.subplots(figsize=(10,3))
# plt.title('Real-time recording of discharges')
plt.xlabel('Time, s')
color = 'C1'
t_0_for_graph = 50
ax1a.set_ylabel('Mirror charge on substrate, nC', color=color)
ax1a.tick_params(axis='y', labelcolor=color)
ax1a.plot(t[delamination_start_index:max_index] - t_0_for_graph,
          filtered[delamination_start_index:max_index], linewidth=0.9,
          color='C1', alpha=0.5)
# ax1a.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1a.xaxis.set_major_locator(loc)
loc2 = plticker.MultipleLocator(base=0.2) # this locator puts ticks at regular intervals
ax1a.xaxis.set_minor_locator(loc2)
dary = filtered[delamination_start_index:max_index] - \
       np.average(filtered[delamination_start_index:max_index])
# steplen = 70
steplen = 20
step = np.hstack((-1*np.ones(steplen), 1*np.ones(steplen)))
dary_step = np.convolve(dary, step, mode='valid')
ax2 = ax1a.twinx()
ax2.tick_params(axis='y', labelcolor='C0')
ax2.set_ylabel('Apparent charge loss, nC', color='C0')
ax2.plot(t[steplen+delamination_start_index:steplen+delamination_start_index + len(dary_step)] - t_0_for_graph,
         dary_step/steplen,
         color='C0', linewidth=0.9)
peaks, _ = find_peaks(dary_step, width=steplen*0.6, prominence=0.25, distance=steplen)
for step_loc in peaks:
    time_here = t[steplen + delamination_start_index + step_loc]
    plt.axvline(x=time_here-t_0_for_graph, color='grey', alpha=0.5)
plt.tight_layout()
fnn.savefig(target_folder+'electrometer/electrometer_discharge_sequence.png', dpi=600)

np.save(target_folder+'electrometer/peaks', peaks)
np.save(target_folder+'electrometer/time', t)
np.save(target_folder+'electrometer/time_for_graph', t[steplen+delamination_start_index:steplen+delamination_start_index + len(dary_step)])
np.save(target_folder+'electrometer/time_for_raw_graph', t[delamination_start_index:max_index])
np.save(target_folder+'electrometer/discharge_magnitude_graph', dary_step/steplen)
np.save(target_folder+'electrometer/charge_graph', filtered[delamination_start_index:max_index])
np.save(target_folder+'electrometer/steplen', steplen)
np.save(target_folder+'electrometer/delamination_start_index', delamination_start_index)

plt.show()
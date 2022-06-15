# -*- coding: utf-8 -*-
"""
Image processing of the motion of detachment front during the delamination
of PDMS stamp from a thin layer of polymer on a substrate.

It also uses data from electrometer connected to the conductive substrate
that measures the charge on this terminal (substrate) synchronously with
camera recording.

Created on Tue Aug 22 01:28:37 2017

@author: Yaroslav I. Sobolev
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy import interpolate
from scipy import signal
# import skimage as sk
# from skimage.morphology import skeletonize
# from skimage import io
# from skimage.color import rgb2hsv
# from skimage.feature import register_translation
# from skimage.transform import AffineTransform, warp, resize
# from skimage import feature
# from skimage.filters import roberts, sobel
import pickle
from lmfit.models import GaussianModel, ConstantModel
from scipy import fftpack
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz
drive_letter = 'Y:'
frames_directory = drive_letter + '\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01\\video'
target_folder = drive_letter + '/PDMS-PMMA_delamination_experiments/data/20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01/'
istart, iend = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\istart_iend.p', "rb"))
front_curves = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\front_curves_spacederiv.p', "rb"))
roi = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\roi.p', "rb"))
pix_per_mm = (334.8-91.2)/9.5
frames_per_sec = 60
stamp_fall_video_frame = 399
c = 0.0033
testing = True

ys_for_psd = np.linspace(74, 125, 20)
xs_for_psd = []
ys_for_splines = np.arange(182)
xs_for_splines = []

# for i, front_curve in enumerate(front_curves[:-10]):
#     # if i > :
#     #     continue
#     xs = front_curve[1,:]
#     ys = front_curve[0,:]
#     plt.plot(xs, ys)
# plt.show()

video_frames_at_discharge = []
video_frames_at_discharge = pickle.load(open(target_folder + 'electrometer/video_frames_at_discharge.pickle', 'rb'))
video_frames_at_discharge = [int(round(x*frames_per_sec)) for x in video_frames_at_discharge]

for i, front_curve in enumerate(front_curves[:-10]):
    # if i > :
    #     continue
    xs = front_curve[1,:]
    ys = front_curve[0,:]
    stdev = np.std(xs)
    median = np.median(xs)
    mask = np.logical_and(xs > median - 100, xs < median + 70)
    xs = np.copy(xs[mask])
    ys = np.copy(ys[mask])
    # low = np.percentile(xs, alpha)
    # high = np.percentile(xs, 100 - alpha)
    alpha = 20
    median = np.median(xs)
    high_std = np.percentile(xs, 100 - alpha) - median
    low_std = median - np.percentile(xs, alpha)
    # mask = np.logical_and(xs > median - 1.1*low_std, xs < median + 1.1*high_std)
    # xs = xs[mask]
    # ys = ys[mask]

    for i in range(40):
        median = np.median(xs)
        dy = np.diff(xs)
        bads = np.where(np.abs(dy) > 10)[0]
        if bads.shape[0] == 0:
            print('it took {0} iteration to cut bads'.format(i))
            break
        bads_for_cutting = []
        for bad in bads:
            distance_to_median_here = np.abs(xs[bad] - median)
            distance_to_median_for_next_element = np.abs(xs[bad+1] - median)
            if distance_to_median_here > distance_to_median_for_next_element:
                bads_for_cutting.append(bad)
            else:
                bads_for_cutting.append(bad+1)
        bads_for_cutting = np.array(list(set(bads_for_cutting)))
        # bads_for_cutting = np.concatenate((bads, (bads + 1)), axis=0) # this was cutting both sides
        ys = np.delete(ys, bads_for_cutting)
        xs = np.delete(xs, bads_for_cutting)
    # mask = np.logical_and(xs < high, xs > low)
    # mask = xs > low
    # plt.hist(xs)
    # plt.show()
    # plt.plot(xs[mask], ys[mask], color='black', alpha=0.5)

    # plt.plot(xs,ys,color='black', alpha=0.5)

    if len(ys) > 5:
        tck = splrep(ys, xs, s=len(ys)*3)
        #    plt.plot(cross_x, cross_y, linewidth=0.8)
        ys2 = np.linspace(min(ys), max(ys), 200)
        xs2 = splev(ys2, tck)
        plt.plot(xs2, ys2, color='black', alpha=0.5)

        x_for_splines = splev(ys_for_splines, tck)
        x_for_psd = np.mean(splev(ys_for_psd, tck))
        xs_for_psd.append(x_for_psd)
        xs_for_splines.append(x_for_splines)

stack_of_splines = np.stack(xs_for_splines)
xs_for_psd = np.array(xs_for_psd)
bkg = np.poly1d(np.polyfit(np.arange(xs_for_psd.shape[0]), xs_for_psd, deg=1))(np.arange(xs_for_psd.shape[0]))
data = xs_for_psd - bkg
f2, ax2 = plt.subplots()
ax2.plot(data)
f3, ax3 = plt.subplots()
rps = 2
ax3.psd(data, len(data) // 1, Fs=rps)
# plt.show()
sig = data
# The FFT of the signal
sig_fft = fftpack.fft(sig)
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(sig.size, d=1 / rps)
# INDUSTRIAL BUTTER FILTER
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandstop', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

filtered_sig = butter_bandpass_filter(data, lowcut=0.24, highcut=0.31, fs=rps, order=5)
# sos1 = signal.butter(10, (58, 62), 'bs', fs=rps, output='sos')
#
# filtered_sig = signal.sosfilt(sos1, sig)
bias_here = 1
ax3.psd(filtered_sig*(1-bias_here) + data*bias_here, len(data) // 1, rps, color='red')
ax2.plot(filtered_sig)

f4 = plt.figure(4)
filtered_sig_raw = butter_bandpass_filter(xs_for_psd, lowcut=0.24, highcut=0.31, fs=rps, order=5)
plt.plot(filtered_sig_raw)
plt.plot(filtered_sig + bkg)
plt.plot(xs_for_psd)

processed_stack_of_splines = np.zeros_like(stack_of_splines)
for i in range(stack_of_splines.shape[1]):
    signal_here = stack_of_splines[:,i]
    bkg_here = np.poly1d(np.polyfit(np.arange(signal_here.shape[0]), signal_here, deg=7))(np.arange(signal_here.shape[0]))
    for_fft_filter = signal_here - bkg_here
    filtered1 =  butter_bandpass_filter(for_fft_filter, lowcut=0.24, highcut=0.31, fs=rps, order=5)
    processed_stack_of_splines[:,i] = filtered1*(1-bias_here) + for_fft_filter*bias_here + bkg_here
    # processed_stack_of_splines[:170,i] = stack_of_splines[:170,i]

f5 = plt.figure(5)
plt.axis('equal')
plt.xlim(0, 750)
to_exclude = [0, 1]
for i in range(processed_stack_of_splines.shape[0]):
    if i in video_frames_at_discharge:
        plt.plot(processed_stack_of_splines[i, :], ys_for_splines, color='green', alpha=0.5, linewidth=3)
f5.savefig(target_folder + 'peeling_fronts_at_discharges.png', dpi=300)
for i in range(processed_stack_of_splines.shape[0]):
    if not (i in to_exclude):
        plt.plot(processed_stack_of_splines[i,:], ys_for_splines, color='black', alpha=0.5, linewidth=0.7)
f5.savefig(target_folder + 'peeling_fronts_positions.png', dpi=300)


f6 = plt.figure(6)#, dpi=200)
average_position_of_front = np.mean(processed_stack_of_splines, axis=1)
plt.plot(average_position_of_front)
    # if not testing:
    #     if k % 10 == 0:
    #         print('Frame {0} is done.'.format(k))
    #         plt.plot(xs, ys, color='blue', linewidth=0.25, alpha=0.5)
front_speed = signal.savgol_filter(average_position_of_front, window_length=9, polyorder=2, deriv=1)
f7,ax7 = plt.subplots()
plt.xlabel('Time (s)')
ax7.tick_params(axis='y', labelcolor='C0')
ax7.set_ylabel('Velocity of delamination (peeling) front (mm/s)', color='C0')
plt.plot(np.arange(front_speed.shape[0])/frames_per_sec, front_speed*frames_per_sec/pix_per_mm)
plt.ylim(2, np.max(front_speed*frames_per_sec/pix_per_mm))


# load electrometer data
peaks = np.load(target_folder + 'electrometer/peaks.npy')
t = np.load(target_folder + 'electrometer/time.npy')
time_for_graph = np.load(target_folder + 'electrometer/time_for_graph.npy')
time_for_raw_graph = np.load(target_folder + 'electrometer/time_for_raw_graph.npy')
discharge_magnitude_graph = np.load(target_folder + 'electrometer/discharge_magnitude_graph.npy')
charge_graph = np.load(target_folder + 'electrometer/charge_graph.npy')
steplen = np.load(target_folder + 'electrometer/steplen.npy', )
delamination_start_index = np.load(target_folder + 'electrometer/delamination_start_index.npy')
delamination_end_index = np.load(target_folder + 'electrometer/delamination_end_index.npy')
t_0_for_graph = np.load(target_folder + 'electrometer/t_0_for_graph.npy')

ax7b = ax7.twinx()
ax7b.tick_params(axis='y', labelcolor='C1')
ax7b.set_ylabel('Apparent charge loss (nC)', color='C1')
# electrometer_minus_video_time_difference = t[delamination_end_index] - front_speed.shape[0]/frames_per_sec
electrometer_minus_video_time_difference = t[delamination_end_index] - (stamp_fall_video_frame - istart)/frames_per_sec
ax7b.plot(time_for_graph - electrometer_minus_video_time_difference, discharge_magnitude_graph, color='C1', linewidth=0.6, alpha=0.7)
video_frames_at_discharge = []
for step_loc in peaks[:-1]:
    time_here = t[steplen + delamination_start_index + step_loc]
    ax7b.axvline(x=time_here - electrometer_minus_video_time_difference, color='grey', alpha=0.7, ymax=0.6)
    video_frames_at_discharge.append(time_here - electrometer_minus_video_time_difference)
pickle.dump(video_frames_at_discharge, open(target_folder + 'electrometer/video_frames_at_discharge.pickle', 'wb+'))

ax7b.set_ylim(-0.2, 0.1)
plt.tight_layout()

f7.savefig(target_folder + 'delamination_speed_vs_electrometer.png', dpi=300)
# f7.savefig(target_folder + 'delamination_speed_vs_electrometer.eps', dpi=300)
plt.show()

import pylab
import imageio
# filename = frames_directory + '\\video.mp4'
# vid = imageio.get_reader(filename,  'ffmpeg')

def load_stabilized_frames(frames_directory):
    print('Loading stabilized frames...')
    frames = np.load(frames_directory + '\\video_stabilized_float.npy')
    first_frame_number = np.load(frames_directory + '\\first_frame.npy')
    print('...Loaded.')
    return frames, first_frame_number

# frames, first_frame_number = load_stabilized_frames()

# plt.show()

# ALL KINDS OF FILTERING
# if len(cross_x) < 5:
#     if not testing:
#         print('Pass on frame {0}'.format(k))
#     continue
# # remove outliers
# dy = np.diff(cross_y)
# bads = np.where(np.abs(dy) > 10)[0]
# bads = np.concatenate((bads, (bads + 1)), axis=0)
# cross_x = np.delete(cross_x, bads)
# cross_y = np.delete(cross_y, bads)
#
# bads = np.where(cross_y > 270)[0]
# #    bads = np.concatenate((bads, (bads+1)), axis=0)
# cross_x = np.delete(cross_x, bads)
# cross_y = np.delete(cross_y, bads)
#
# tck = splrep(cross_x, cross_y, s=len(cross_x))
# #    plt.plot(cross_x, cross_y, linewidth=0.8)
# xs = np.linspace(min(cross_x), max(cross_x), 200)
# ys = splev(xs, tck)
# if not testing:
#     if k % 10 == 0:
#         print('Frame {0} is done.'.format(k))
#         plt.plot(xs, ys, color='blue', linewidth=0.25, alpha=0.5)
#
# if not testing:
#     plt.xlim([0, 300])
#     plt.ylim([0, 300])
#     plt.savefig('front_family.eps', dpi=800)
#     plt.savefig('front_family.png', dpi=800)

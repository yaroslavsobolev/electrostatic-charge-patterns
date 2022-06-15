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
from skimage import io
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

frames_directory = 'experimental_data\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01\\video'
target_folder = 'experimental_data/PDMS-PMMA_delamination_experiments/data/20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01/'
istart, iend = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\istart_iend.p', "rb"))
front_curves = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\front_curves_spacederiv.p', "rb"))
roi = pickle.load(open(frames_directory + '\\delamination_fronts\\fronts\\roi.p', "rb"))
pix_per_mm = (334.8-91.2)/9.5
frames_per_sec = 60
stamp_fall_video_frame = 399
c = 0.0033
testing = True

pts = [[370, 1240],
       [348, 168],
       [212, -50],
       [114, 160],
       [145, 1245]]
pts.append(pts[0])
pts = np.array(pts)
# roi = [142, 337, 385, 1192]


top_edge = np.poly1d(np.polyfit([pts[0, 1], pts[1, 1]], [pts[0, 0], pts[1, 0]], 1))
bottom_edge = np.poly1d(np.polyfit([pts[3, 1], pts[4, 1]], [pts[3, 0], pts[4, 0]], 1))


ys_for_psd = np.linspace(74, 125, 20)
xs_for_psd = []
# ys_for_splines = np.arange(182)
ys_for_splines = np.arange(-30, 230)
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
processed_stack_of_splines = stack_of_splines

# ==================== SHOW FRONT POSITIONS
f5, ax5 = plt.subplots()
ax5.set_aspect('equal')
ax5.set_xlim(-60, 1250)
to_exclude = [0, 1]

skp_map_base = io.imread(target_folder + '/video/skp_synth/skp_frame_base.png')
skp_map_blank = io.imread(target_folder + '/video/skp_synth/skp_frame_blank.png')
ax5.imshow(skp_map_blank)
# ax5.plot(pts[:,1], 719-pts[:,0])
ax5.plot(pts[:,1], pts[:,0], color='darkorchid')
# plt.show()

for i in range(processed_stack_of_splines.shape[0]):
    print(i)
    if not (i in to_exclude):
        # xs_for_plotting = roi[2] + processed_stack_of_splines[i,:]
        # ys_for_plotting = roi[0]+ys_for_splines
        # k_max = 0
        xs_for_plotting = []
        ys_for_plotting = []
        for k, x_here in enumerate(roi[2] + processed_stack_of_splines[i,:]):
            if top_edge(x_here) > roi[0]+ys_for_splines[k] and \
                    bottom_edge(x_here) < roi[0]+ys_for_splines[k]:
                # plt.scatter(x_here, top_edge(x_here))
                # plt.scatter(x_here, bottom_edge(x_here))
                xs_for_plotting.append(x_here)
                ys_for_plotting.append(roi[0]+ys_for_splines[k])

        ax5.plot(xs_for_plotting, ys_for_plotting, color='black', alpha=0.5, linewidth=0.35)
        if i in video_frames_at_discharge:
            ax5.plot(xs_for_plotting, ys_for_plotting, color='green', alpha=0.5, linewidth=3)
    ax5.set_xlim(-60, 1250)
    f5.savefig(target_folder + 'video/discharges/frames1/{0:04d}.png'.format(i), dpi=300)

    # f5.savefig(target_folder + 'peeling_fronts_positions.png', dpi=300)

plt.show()

# # =============================== Reconstructing SKP map evolution
# skp_map_base = io.imread(target_folder + '/video/skp_synth/skp_frame_base.png')
# skp_map_blank = io.imread(target_folder + '/video/skp_synth/skp_frame_blank.png')
# skp_map_now = io.imread(target_folder + '/video/skp_synth/skp_frame_00.png')
# # plt.show()
# current_spark = 0
#
# for i in range(processed_stack_of_splines.shape[0]):
#     print(i)
#     f5, ax5 = plt.subplots()
#     ax5.set_aspect('equal')
#     ax5.set_xlim(-60, 1250)
#     to_exclude = [0, 1]
#     ax5.imshow(skp_map_now)
#     # ax5.plot(pts[:,1], 719-pts[:,0])
#     ax5.plot(pts[:, 1], pts[:, 0], color='darkorchid')
#     if not (i in to_exclude):
#         # xs_for_plotting = roi[2] + processed_stack_of_splines[i,:]
#         # ys_for_plotting = roi[0]+ys_for_splines
#         # k_max = 0
#         xs_for_plotting = roi[2] + processed_stack_of_splines[i,:]
#         ys_for_plotting = roi[0]+ys_for_splines
#         # for k, x_here in enumerate(roi[2] + processed_stack_of_splines[i,:]):
#         #     if top_edge(x_here) > roi[0]+ys_for_splines[k] and \
#         #             bottom_edge(x_here) < roi[0]+ys_for_splines[k]:
#         #         # plt.scatter(x_here, top_edge(x_here))
#         #         # plt.scatter(x_here, bottom_edge(x_here))
#         #         xs_for_plotting.append(x_here)
#         #         ys_for_plotting.append(roi[0]+ys_for_splines[k])
#
#         ax5.fill_betweenx(y=ys_for_plotting, x1=xs_for_plotting, x2=1245*np.ones_like(xs_for_plotting),
#                           color='white', alpha=1)
#         if i in video_frames_at_discharge:
#             # ax5.plot(xs_for_plotting, ys_for_plotting, color='green', alpha=0.5, linewidth=3)
#             current_spark = video_frames_at_discharge.index(i)
#             skp_map_now = io.imread(target_folder + '/video/skp_synth/skp_frame_{0:02d}.png'.format(current_spark))
#             ax5.imshow(skp_map_now)
#             print('New cur spark:{0}'.format(current_spark))
#
#     # ax5.set_xlim(0, 750)
#     f5.savefig(target_folder + 'video/discharges/frames2/{0:04d}.png'.format(i), dpi=300)
#     plt.close(f5)
#
# plt.show()





# # =============== plotting electrometer
# # f6 = plt.figure(6)
# average_position_of_front = np.mean(processed_stack_of_splines, axis=1)
# # plt.plot(average_position_of_front)
#     # if not testing:
#     #     if k % 10 == 0:
#     #         print('Frame {0} is done.'.format(k))
#     #         plt.plot(xs, ys, color='blue', linewidth=0.25, alpha=0.5)
# front_speed_0 = signal.savgol_filter(average_position_of_front, window_length=9, polyorder=2, deriv=1)
#
#
# # load electrometer data
# peaks = np.load(target_folder + 'electrometer/peaks.npy')
# t = np.load(target_folder + 'electrometer/time.npy')
# time_for_graph = np.load(target_folder + 'electrometer/time_for_graph.npy')
# time_for_raw_graph = np.load(target_folder + 'electrometer/time_for_raw_graph.npy')
# discharge_magnitude_graph = np.load(target_folder + 'electrometer/discharge_magnitude_graph.npy')
# charge_graph = np.load(target_folder + 'electrometer/charge_graph.npy')
# steplen = np.load(target_folder + 'electrometer/steplen.npy', )
# delamination_start_index = np.load(target_folder + 'electrometer/delamination_start_index.npy')
# delamination_end_index = np.load(target_folder + 'electrometer/delamination_end_index.npy')
# t_0_for_graph = np.load(target_folder + 'electrometer/t_0_for_graph.npy')
#
# for frame_id in range(1, front_speed_0.shape[0]):
#     print(frame_id)
#     front_speed = front_speed_0[:frame_id]
#     f7,ax7 = plt.subplots(figsize=(5,4))
#     plt.xlabel('Time (s)')
#     ax7.tick_params(axis='y', labelcolor='C0')
#     ax7.set_ylabel('Velocity of delamination (peeling) front (mm/s)', color='C0')
#     ax7.plot(np.arange(front_speed.shape[0])/frames_per_sec, front_speed*frames_per_sec/pix_per_mm)
#     speed_scat = ax7.scatter([(front_speed.shape[0]-1)/frames_per_sec],
#                              [front_speed[-1]*frames_per_sec/pix_per_mm], color='C0')
#     ax7.set_ylim(2, np.max(front_speed_0*frames_per_sec/pix_per_mm))
#     ax7b = ax7.twinx()
#     ax7b.tick_params(axis='y', labelcolor='C1')
#     ax7b.set_ylabel('Apparent charge loss (nC)', color='C1')
#     # electrometer_minus_video_time_difference = t[delamination_end_index] - front_speed.shape[0]/frames_per_sec
#     electrometer_minus_video_time_difference = t[delamination_end_index] - (stamp_fall_video_frame - istart)/frames_per_sec
#     mask = (time_for_graph - electrometer_minus_video_time_difference) < (frame_id-1)/frames_per_sec
#     times_here = time_for_graph - electrometer_minus_video_time_difference
#     ax7b.plot(times_here[mask], discharge_magnitude_graph[mask], color='C1', linewidth=0.6, alpha=0.7)
#     ax7b.scatter([times_here[mask][-1]], [discharge_magnitude_graph[mask][-1]], color='C1')
#     # video_frames_at_discharge = []
#     for step_loc in peaks[:-1]:
#         time_here = t[steplen + delamination_start_index + step_loc]
#         ax7b.axvline(x=time_here - electrometer_minus_video_time_difference, color='grey', alpha=0.7, ymax=0.6)
#     #     video_frames_at_discharge.append(time_here - electrometer_minus_video_time_difference)
#     # pickle.dump(video_frames_at_discharge, open(target_folder + 'electrometer/video_frames_at_discharge.pickle', 'wb+'))
#
#     ax7b.set_ylim(-0.2, 0.1)
#     ax7b.set_xlim(-1, 6)
#     # plt.tight_layout()
#     # if frame_id == 1:
#     f7.tight_layout()
#     f7.savefig(target_folder + '/video/electrometer_frames/frames/frame{0:04d}'.format(frame_id), dpi=300)
#     # plt.show()
#     plt.close(f7)


# # ============================ make pull force plot
# adhesion_t = np.load(target_folder + 'force/force_and_coulombmeter_vs_time_t.npy')
# adhesion_f = np.load(target_folder + 'force/force_and_coulombmeter_vs_time_f.npy')
# f = interpolate.interp1d(adhesion_t, adhesion_f)
# t_new = np.linspace(np.min(adhesion_t), np.max(adhesion_t), 10000)
# f_new = f(t_new)
# for frame_id in range(1, 320):
#     print(frame_id)
#     # front_speed = front_speed_0[:frame_id]
#     f7,ax7 = plt.subplots(figsize=(5,4))
#     plt.xlabel('Time (s)')
#     # ax7.tick_params(axis='y', labelcolor='C0')
#     ax7.set_ylabel('Pull force, N')
#     mask = adhesion_t < (frame_id - 1) / frames_per_sec
#     ax7.plot(adhesion_t[mask], adhesion_f[mask], 'x', color='black')
#     mask2 = t_new < (frame_id - 1) / frames_per_sec
#     ax7.plot(t_new[mask2], f_new[mask2], color='black', alpha=0.5)
#     ax7.scatter(t_new[mask2][-1], f_new[mask2][-1], color='black')
#     # ax7.plot(adhesion_t[mask], adhesion_f[mask])
#     ax7.set_ylim(-0.05, 1.4)
#     ax7.set_xlim(-1, 6)
#     # plt.tight_layout()
#     # if frame_id == 1:
#     f7.tight_layout()
#     f7.savefig(target_folder + '/video/force_frames/frame{0:04d}'.format(frame_id), dpi=300)
#     # plt.show()
#     plt.close(f7)


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

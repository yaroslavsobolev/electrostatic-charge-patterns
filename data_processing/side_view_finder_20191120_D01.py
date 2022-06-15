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

import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy import interpolate
from scipy import signal
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage import io
from skimage.color import rgb2hsv
from skimage.feature import register_translation
from skimage.transform import AffineTransform, warp, resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import feature
from skimage import draw
from skimage.filters import roberts, sobel
from scipy.signal import find_peaks
import pickle
import time
t0 = time.time()
from lmfit.models import GaussianModel, ConstantModel

def shift_image(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)
    return shifted

frames_directory = 'Y:\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01\\video'
istart = 81 #int(round(57*30))+5+20
iend = 401  #2280
c = 0.0033
deriv_step = 2
testing = True

# side view dimensions
stamp_length_in_px = 1190.1
stamp_height_in_px = 98
stamp_width_in_px = 135.6
stamp_length_in_mm = 51.4
stamp_height_in_mm = 5
stamp_width_in_mm = 9
viewing_angle_1 = np.arctan(stamp_width_in_px/stamp_height_in_px * stamp_height_in_mm/stamp_width_in_mm)
viewing_angle_2 = np.arccos(stamp_height_in_px / (stamp_length_in_px / stamp_length_in_mm) / stamp_height_in_mm)
print(viewing_angle_1/np.pi*180)
print(viewing_angle_2/np.pi*180)
viewing_angle = 0.5*(viewing_angle_1 + viewing_angle_2)
import pylab
import imageio
filename = frames_directory + '\\video.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

def load_stabilized_frames(frames_directory):
    print('Loading stabilized frames...')
    frames = np.load(frames_directory + '\\video_stabilized_float.npy')
    first_frame_number = np.load(frames_directory + '\\first_frame.npy')
    print('...Loaded.')
    return frames, first_frame_number

frames, first_frame_number = load_stabilized_frames(frames_directory)

# from skimage.draw import polygon

plt.imshow(frames[0,:,:])
pts = [[370, 1240],
       [348, 168],
       [212, -50],
       [114, 160],
       [145, 1245]]
pts.append(pts[0])
polygon = pts
pts = np.array(pts)
img = np.copy(frames[0,:,:])
rr, cc = draw.polygon(pts[:, 0], pts[:, 1], img.shape)
img[rr, cc] = 1
io.imsave(frames_directory + '\\first_frame_stamp_location.png', img)
plt.imshow(img)

# plt.plot(pts[:,1], pts[:,0], color='red')

# for i in range(0, 20):
#     plt.imshow(frames[-25+i, :, :])
#     plt.show()
plt.show()

# interface_line = np.array([[517.109, 651.388], [1211, 660]])
interface_line = np.array([[517.109, 651.388], [1256, 664]])
# def distance_to_interface = d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
def get_inteface_y(x):
    return interface_line[0, 1] + (x - interface_line[0, 0])/(interface_line[1, 0] - interface_line[0, 0])*\
           (interface_line[1, 1]-interface_line[0, 1])

previous_front_x = 468
previous_gap = 643

# previous_front_x = 1165
# previous_gap = 588

front_search_window = [-10, 40]
gap_search_window = [-7, 5]
front_positions_x = []
angles = []
gaps_y = []
gaps = []
gap_x = 310
n_frames = 317
for frame_id in range(0,n_frames):
    print('Processing frame {0}'.format(frame_id))
    image_for_showing = np.copy(frames[frame_id])
    fig, ax = plt.subplots()
    window_vmin = 615
    window_vmax = 688
    # ax.imshow(image_for_showing, cmap=plt.cm.gray)
    xs = list(range(0, 1279))
    signal = []
    strip_width = 10
    for x in xs:
        y = int(round(get_inteface_y(x)))
        # ax.scatter(x, y, color='yellow', alpha=0.5)
        signal.append(np.mean(image_for_showing[y:y+strip_width, x]))
        # signal.append(image_for_showing[x, y])
    plt.plot([0, xs[-1]], [get_inteface_y(0), get_inteface_y(xs[-1])], color='yellow', alpha=0.2)
    fig2,axarr = plt.subplots(3,1)#, sharex=True)
    axarr[0].plot(xs, signal)
    axarr[1].plot(xs[:-1], np.diff(signal, n=1))
    deriv = -1*savgol_filter(np.diff(signal, n=1), 15, 2)
    axarr[2].plot(xs[:-1], deriv)
    peaks, _ = find_peaks(deriv, distance=50, prominence=1.7e-9)
    axarr[2].plot(peaks, deriv[peaks], "x")
    ax.imshow(image_for_showing, cmap=plt.cm.gray)
    # plt.show()
    print('Peaks: {0}'.format(peaks))
    new_front_position = peaks[np.logical_and(peaks > previous_front_x + front_search_window[0],
                         peaks < previous_front_x + front_search_window[1])][0]
    print('New front x: {0}'.format(new_front_position))
    previous_front_x = new_front_position
    front_positions_x.append(new_front_position)
    ax.scatter(new_front_position, get_inteface_y(new_front_position), marker='x', color='red',alpha=0.3)

    ys = list(range(0, image_for_showing.shape[0]))
    signal2 = []
    front_search_window = [-10, 10]
    for y in ys:
        signal2 = image_for_showing[:, gap_x]
    fig3, axarr3 = plt.subplots(3,1, sharex=True)
    axarr3[0].plot(signal2)
    axarr3[1].plot(ys[:-1], np.diff(signal2, n=1))
    deriv = savgol_filter(np.diff(signal2, n=1), 15, 2)
    axarr3[2].plot(ys[:-1], deriv)
    peaks, _ = find_peaks(deriv, distance=13, prominence=1.7e-9)
    axarr3[2].plot(peaks, deriv[peaks], "x")
    # ax.imshow(image_for_showing, cmap=plt.cm.gray)
    # plt.show()
    print('Peaks: {0}'.format(peaks))
    print('Interface_y: {0}'.format(get_inteface_y(gap_x)))
    new_gap = peaks[np.logical_and(peaks < get_inteface_y(gap_x),
                         np.logical_and(peaks > previous_gap + gap_search_window[0],
                                        peaks < previous_gap + gap_search_window[1]))][0]
    print('New gap y: {0}'.format(new_gap))
    previous_gap = new_gap
    gaps_y.append(new_gap)
    gaps.append(get_inteface_y(gap_x)-new_gap)
    angle_here = np.arctan( (get_inteface_y(gap_x)-new_gap)/np.cos(viewing_angle)/(new_front_position - gap_x) )
    angles.append(angle_here)
    ax.scatter(gap_x, new_gap, marker='x', color='red',alpha=0.3)

    if frame_id == 317:
        gap_x = 380
        previous_gap = 594



    # image = frames[frame_id,window_vmin:window_vmax,:] > 9e-8
    # coords = corner_peaks(corner_harris(image), min_distance=10, threshold_rel=0.04)
    # coords_subpix = corner_subpix(image, coords, window_size=13)
    # ax.plot(coords[:, 1], coords[:, 0]+window_vmin, color='cyan', marker='o',
    #         linestyle='None', markersize=6)
    # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0]+window_vmin, '+r', markersize=15)
    fig.savefig(frames_directory + '\\gaps\\processing_frames\\{0:05d}.png'.format(frame_id))
    fig2.savefig(frames_directory + '\\gaps\\front_positions\\{0:05d}.png'.format(frame_id))
    fig3.savefig(frames_directory + '\\gaps\\gap_positions\\{0:05d}.png'.format(frame_id))
    # plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)

np.save(frames_directory + '\\gaps\\front_positions_x', np.array(front_positions_x))
np.save(frames_directory + '\\gaps\\angles', np.array(angles))
np.save(frames_directory + '\\gaps\\gaps_y', np.array(gaps_y))
np.save(frames_directory + '\\gaps\\gaps', np.array(gaps))
print('Done')

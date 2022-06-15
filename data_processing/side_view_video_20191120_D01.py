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
from skimage.filters import roberts, sobel
from scipy.signal import find_peaks
from matplotlib_scalebar.scalebar import ScaleBar
import pickle
import time
t0 = time.time()
from lmfit.models import GaussianModel, ConstantModel
import pylab
import imageio
frames_per_sec = 60
target_folder = 'Y:\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01'
video_frames_at_discharge = []
video_frames_at_discharge = pickle.load(open(target_folder + '\\electrometer\\video_frames_at_discharge.pickle', 'rb'))
video_frames_at_discharge = [int(round(x*frames_per_sec)) for x in video_frames_at_discharge]

frames_directory = target_folder + '\\video'
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
mm_in_px = stamp_length_in_mm/stamp_length_in_px

interface_line = np.array([[517.109, 651.388], [1256, 664]])
# def distance_to_interface = d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
def get_inteface_y(x):
    return interface_line[0, 1] + (x - interface_line[0, 0])/(interface_line[1, 0] - interface_line[0, 0])*\
           (interface_line[1, 1]-interface_line[0, 1])

previous_front_x = 468
previous_gap = 643

front_search_window = [-10, 40]
gap_search_window = [-7, 5]
front_positions_x = []
angles = []
gaps_y = []
gaps = []
gap_x = 310
n_frames = 317

front_positions_x = np.load(frames_directory + '\\gaps\\front_positions_x.npy')
angles = np.load(frames_directory + '\\gaps\\angles.npy')
gaps_y = np.load(frames_directory + '\\gaps\\gaps_y.npy')
gaps = np.load(frames_directory + '\\gaps\\gaps.npy')

xs = np.arange(angles.shape[0])
spl = UnivariateSpline(xs, savgol_filter(angles, 41, 2), s=0.00002)
# spl.set_smoothing_factor(3)
# plt.plot(angles, 'o')
# plt.plot(savgol_filter(angles, 41, 2))
# plt.plot(xs, spl(xs))
# plt.show()
angles = spl(xs)
previous_front_position = -37/mm_in_px

for frame_id in range(0,n_frames):
    fig, ax = plt.subplots()
    print('Processing frame {0}'.format(frame_id))
    ax.axhspan(-5, 0, color='C0', alpha=0.5)
    x_f = front_positions_x[frame_id]
    y_f = get_inteface_y(x_f)
    x_f_from_right = np.linalg.norm( np.array([x_f, y_f]) - np.array([interface_line[1]]) )
    from_front_to_left = stamp_length_in_px - x_f_from_right
    angle = angles[frame_id]
    left_bottom_corner = [-x_f_from_right - from_front_to_left*np.cos(angle), from_front_to_left*np.sin(angle)]
    points = [[0, 0],
              [-x_f_from_right, 0],
              left_bottom_corner,
              [left_bottom_corner[0] + stamp_height_in_mm/mm_in_px*np.sin(angle),
                    left_bottom_corner[1] + stamp_height_in_mm/mm_in_px*np.cos(angle)],
              [-x_f_from_right, stamp_height_in_mm/mm_in_px],
              [0, stamp_height_in_mm/mm_in_px]]
    points = np.array(points)
    points = points*mm_in_px
    ax.fill(points[:,0], points[:,1], color='yellowgreen', alpha=0.7)
    ax.scatter(points[1, 0], points[1, 1], color='black', alpha=1, zorder=100)
    if frame_id in video_frames_at_discharge:
        points_spark = [[-x_f_from_right, 0],
                        [previous_front_position, 0],
                        [previous_front_position, (-x_f_from_right-previous_front_position)*np.sin(angle)]]
        points_spark = np.array(points_spark)*mm_in_px
        ax.fill(points_spark[:, 0], points_spark[:, 1], color='darkorchid', alpha=1)
        previous_front_position = -x_f_from_right
        # plt.show()

    ax.set_aspect('equal')
    scalebar = ScaleBar(dx=1, units='mm')  # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    ax.set_ylim(-5, 12)
    ax.set_xlim(-stamp_length_in_mm - 5, 5)

    ax.text(x=points[1, 0], y=points[1, 1]+1, s='Peeling front')
    ax.text(x=points[2, 0]+1, y=points[2, 1]+1, s='Stamp (PDMS)')
    ax.text(x=-stamp_length_in_mm-1, y=-4, s='Wafer with PMMA film')
    fig.savefig(frames_directory + '\\gaps\\side_view_frames\\{0:04}.png'.format(frame_id), dpi=200)
    # plt.show()
    plt.close(fig)
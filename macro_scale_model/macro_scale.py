import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy import interpolate
from scipy.interpolate import Rbf
import pyautogui
import time
import os.path
import os
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import matplotlib.ticker as mticker
f_mticker = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g_scinotation = lambda x,pos : "${}$".format(f_mticker._formatSciNotation('%1.10e' % x))

buttons = dict()

# # COORDS ON MY PC
# buttons['params'] = (267, 521)
# buttons['expression'] = (1000, 735)
# buttons['expr_box'] = (817, 1466)
# buttons['compute'] = (1647, 146)
# buttons['blank'] = (3080, 1944)
# figsize = (16, 12)

# COORDS ON LAB PC
buttons['params'] = (1094, 272)
buttons['expression'] = (1321, 425)
buttons['expr_box'] = (1372, 740)
buttons['compute'] = (1581, 68)
buttons['blank'] = (1839, 948)
figsize = (8, 6)

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def comsol_click(button_name):
    time.sleep(0.3)
    pyautogui.click(buttons[button_name])

def comsol_change_params_and_compute(stamp_coverage=False):
    comsol_click('blank')
    comsol_click('blank')
    while True:
        found_params = pyautogui.locateOnScreen('puautogui_pics/parameters.png')
        if found_params:
            break
        else:
            logger.debug('Did not find params button. Comsol is doing smth. Waiting 2 sec.')
            time.sleep(2)
    if (not (type(stamp_coverage) is bool)):
        comsol_click('params')
        time.sleep(0.6)
        comsol_click('expression')
        time.sleep(0.6)
        comsol_click('expr_box')
        time.sleep(0.6)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.typewrite(str(stamp_coverage))
    comsol_click('compute')

# load breakdown field spline interpolator

#AIR CASE
# paschen_spline = pickle.load(open("paschen_air.pickle", "rb"))

# ARGON CASE
paschen_spline = pickle.load(open("paschen_air.pickle", "rb"))

def breakdown_field_at_gap(gap_in_m):
    gap_in_mm = gap_in_m*1e3
    result = np.power(10, interpolate.splev(np.log10(gap_in_mm), paschen_spline))
    # result[gap_in_m <= 0] = 1e20
    return result*1e6 # conversion from kV/mm to V/m

# # test Paschen curve
# f_pash, ax = plt.subplots(1, figsize=(11,7.5))
# plt.xscale('log')
# plt.yscale('log')
# xs = np.logspace(-4.5, 3, 1000)
# ax.plot(xs, breakdown_field_at_gap(xs), 'grey', lw=3, alpha=0.3, zorder=-8,
#         label='Empirical interpolating function')
# plt.ylabel('Breakdown field, kV/mm')
# plt.xlabel('Gap between electrodes, mm')
# plt.legend()
# plt.ylim([1, 5000])
# plt.show()

iter_id = 1
sample_dimension = 50
number_of_charge_points = 500
prederjaguin_charge_density = 9 # nC/cm^2
trajectory_length_cutoff = 20 # in mm

# TODO: Make the charge_points_x coordinates follow the stamp as "angle" increases -- this would be realistic.
#  Having static x values is not. This should be changed both here and in COMSOL, consistently.
def get_charge_points_x(stamp_coverage, angle, npoints=number_of_charge_points):
    length_of_delaminated_section = sample_dimension * (1 - stamp_coverage)
    delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
    stamp_end_x = delam_point_x + length_of_delaminated_section*np.cos(angle)
    projected_stamp_length = sample_dimension*np.cos(angle)
    return np.linspace(stamp_end_x-projected_stamp_length, stamp_end_x, npoints)

charge_points_x = get_charge_points_x(0.99, 10 * (np.pi / 180), npoints=number_of_charge_points) # in mm

charge_density = prederjaguin_charge_density*(1e-9)/(1e-2)**2*np.ones_like(charge_points_x) # in C/m^2
stamp_coverage = 0.99
angle = 10 * (np.pi/180)

def find_discharges(input_charge_profile, stamp_coverage, iter_id, target_folder, axpasch, ax_illustr, ax_illustr_zoom,
                    old_discharge_locations, to_run_in_meantime):
    sample_dimension = 50
    delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
    angle = 10 * (np.pi / 180)
    def x_to_y_on_stamp_bottom_line(x):
        angle = 10 * (np.pi / 180)
        delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
        return (x - delam_point_x) * np.tan(angle)

    logger = logging.getLogger('main_application.find_discharges')
    # simulation_charge_density = 1e-9/((1e-2)**2) # coulombs per meter square
    # export charge profile to .txt for COMSOL to use
    for_comsol = np.vstack((charge_points_x, input_charge_profile)).T
    np.savetxt('comsol/table.txt',
               for_comsol,
               delimiter=' ')
    time.sleep(1)
    # remember modified date of fields file
    fields_filename = 'comsol/test.txt'
    init_mdate = time.ctime(os.path.getmtime(fields_filename))
    # compute
    while True:
        comsol_change_params_and_compute(stamp_coverage=stamp_coverage)
        t0_comsol = time.time()
        meantime_was_done = False
        for i in range(30):
            if not meantime_was_done:
                to_run_in_meantime()
                meantime_was_done = True
            else:
                logger.debug('Waiting...')
                plt.pause(0.3)
            # TODO: Spend this waiting time on saving the plot figures on hard drive, instead of just waiting
            if time.ctime(os.path.getmtime(fields_filename)) != init_mdate and \
                    os.path.getsize(fields_filename) > 1000000:
                logger.debug('It took COMSOL {0:.2f} seconds to finish. File size: {1}'.format(time.time()-t0_comsol, os.path.getsize(fields_filename)))
                break
        time.sleep(0.2) # Small delay just in case filesystem does lag
        # make sure that the stamp_coverage had the intended value
        while True:
            global_params_from_comsol = np.loadtxt('comsol/global_variable_probes.txt', skiprows=5)
            if len(global_params_from_comsol):
                break
            else:
                logger.debug('Failed to load global param_file. Retrying...')
                time.sleep(3)
        if np.isclose(global_params_from_comsol[2], stamp_coverage, atol=0.00001):
            break
        else:
            logger.debug('Failed consistency check for stamp_coverage from COMSOL. Retrying...')
            time.sleep(3)
    # print('loading comsol txt')
    field_data = np.loadtxt(fields_filename, skiprows=9)
    air_rows = field_data[:,5] == 1
    # print('loaded comsol txt')
    x = field_data[:,0]
    y = field_data[:,1]
    # The following long messy procedure just removes all the PDMS domain points that are on the PDMS lower surface that
    # has delaminated already
    mask_of_bnd3 = field_data[:,5] == 3
    mask_of_boundary = np.logical_and(np.isclose( (y/(x - delam_point_x)), np.tan(angle), atol = 0.001),
                                      x < sample_dimension/2*1.3)
    points_of_pdms_boundary = np.logical_and(mask_of_bnd3, mask_of_boundary)
    # Similarly, remove horiaontal PMMA/air boundary from the data
    mask_of_bnd2 = field_data[:,5] == 2
    mask_of_boundary = np.logical_and(np.isclose(y, 0, atol = 0.001),
                                      x < sample_dimension/2*1.3)
    points_of_pmma_boundary = np.logical_and(mask_of_bnd2, mask_of_boundary)
    minus_boundaries_mask = np.invert(np.logical_or(points_of_pdms_boundary, points_of_pmma_boundary))

    xmax_here = sample_dimension/2 + (sample_dimension/2-delam_point_x)*0.2
    interp_box = {'x':[delam_point_x-(1e-5), xmax_here],
                  'y':[-1e-6, x_to_y_on_stamp_bottom_line(xmax_here)]}
    interp_nsteps = {'x': 1600, 'y': 900}
    box_mask_x = np.logical_and(x > interp_box['x'][0], x < interp_box['x'][1])
    box_mask_y = np.logical_and(y > interp_box['y'][0], y < interp_box['y'][1])
    box_mask = np.logical_and(box_mask_x, box_mask_y)

    final_mask = np.logical_and(box_mask, minus_boundaries_mask)
    x = x[final_mask]
    y = y[final_mask]
    Ex_raw = field_data[final_mask,2]
    Ey_raw = field_data[final_mask,3]



    # # Plotting points of the mesh for debugging
    # mask1 = np.logical_and(x > interp_box['x'][0], x < interp_box['x'][1])
    # mask2 = np.logical_and(y > interp_box['y'][0], y < interp_box['y'][1])
    # mask_combined = np.logical_and(mask1, mask2)
    # ax_illustr_zoom.scatter(x[mask_combined], y[mask_combined], color='red', s=0.2, alpha=0.5)

    xs = [delam_point_x, 0.5 * sample_dimension]
    ax_illustr_zoom.plot(xs, [x_to_y_on_stamp_bottom_line(x) for x in xs], color='black', linewidth=0.5)
    # plt.show()

    # OLD INTERPOLATION ROUTINE
    xi = np.linspace(interp_box['x'][0],interp_box['x'][1],interp_nsteps['x'])
    yi = np.linspace(interp_box['y'][0],interp_box['y'][1],interp_nsteps['y'])
    X, Y = np.meshgrid(xi,yi)

    # nparam = 1
    # param = param_values[nparam]
    # Ex0 = field_data[:,2]
    Ex = griddata((x, y), Ex_raw, (X,Y), method='linear')
    Ey = griddata((x, y), Ey_raw, (X,Y), method='linear')
    Ex_interp = RegularGridInterpolator(points=[xi, yi], values=Ex.T, method='linear')
    Ey_interp = RegularGridInterpolator(points=[xi, yi], values=Ey.T, method='linear')
    # print('interpolated')

    def field_at_points(points):
        return np.stack((Ex_interp(points),Ey_interp(points))).T
    # END OF OLD INTERPOLATION ROUTINE

    # ## ALTERNATIVE INTERPOLATOR
    # xmax_here = sample_dimension/2
    # interp_box = {'x':[delam_point_x, xmax_here],
    #               'y':[-1e-6, x_to_y_on_stamp_bottom_line(xmax_here)]}
    # mask1 = np.logical_and(x > interp_box['x'][0], x < interp_box['x'][1])
    # mask2 = np.logical_and(y > interp_box['y'][0], y < interp_box['y'][1])
    # mask_combined = np.logical_and(mask1, mask2)
    # x_in_box = x[mask_combined]
    # y_in_box = y[mask_combined]
    # # plt.scatter(x_in_box,y_in_box,alpha=0.3)
    # # plt.show()
    # rbf_Ex = Rbf(x[mask_combined], y[mask_combined], Ex_raw[mask_combined])
    # rbf_Ey = Rbf(x[mask_combined], y[mask_combined], Ey_raw[mask_combined])
    # def field_at_points(points):
    #     res = np.stack((rbf_Ex(points[0], points[1]),rbf_Ey(points[0], points[1]))).T
    #     return [res]

    def air_boundary(stamp_coverage, ax):
        sample_dimension = 50
        delam_point_x = sample_dimension*(0.5-(1-stamp_coverage))
        angle = 10 * (np.pi/180)
        # ws = 6e-3
        # ds = 7e-3
        # hs = 1e-3
        # stamp_feature_height = 5e-3
        boundary = Polygon(((delam_point_x, 0),
                            (sample_dimension,    0),
                            (sample_dimension, (sample_dimension-delam_point_x)*np.tan(angle))
                           ))
        x_, y_ = boundary.exterior.xy
        ax.plot(x_, y_, color='grey')
        return boundary

    def draw_geometry(ax):
        linewidth = 0.2
        delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
        xs = [-0.5*sample_dimension, 0.5*sample_dimension]
        ax.plot(xs, [0,0], color='black', linewidth=linewidth)
        length_of_delaminated_section = sample_dimension * (1 - stamp_coverage)
        delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
        stamp_end_x = delam_point_x + length_of_delaminated_section * np.cos(angle)
        xs = [delam_point_x, stamp_end_x]
        ax.plot(xs, [x_to_y_on_stamp_bottom_line(x) for x in xs], color='black', linewidth=linewidth)
        ax.set_xlim([-0.5*sample_dimension, 0.6*sample_dimension])
        ax.set_ylim([-0.1, sample_dimension*np.sin(angle)])

    def process_trajectory(starting_point, folder_for_datasets, breakdown_field_correction=1):
        # testing direction
        # ax_illustr.quiver(starting_point[0], starting_point[1],
        #                 1e-3*initial_surface_tangential_vector[0],
        #                 1e-3*initial_surface_tangential_vector[1], color = 'r')
        traj_is_sparking = False
        dt = 0.0005
        pos = starting_point
        trajectory = [np.copy(pos)]
        for plot_ax in [ax_illustr, ax_illustr_zoom]:
            plot_ax.scatter(starting_point[0], starting_point[1], color='green', s=0.3)
        draw_geometry(ax_illustr)
        length = 0
        Emax = 0
        ythresh = 0
        # First, we should figure out whether the positive or negative direction of intergrating along the field is the
        # outward trajectory (going away from the starting surface, that is). We do this by computing the angle
        # between the direction of the field at the starting point and the tangent vector of the surface.
        # This must be a three-dimensional vector.
        # For current geometry, the surface is flat, so that vector is same at all points.
        initial_surface_tangential_vector = np.array([np.cos(angle), np.sin(angle), 0])
        # taking vector product
        E_here = -1 * field_at_points(pos)[0]
        if np.cross(np.array([E_here[0], E_here[1], 0]), initial_surface_tangential_vector)[2] > 0:
            electrons_move_outward = True
            field_direction = 1
            # logger.debug('Electrond move outward.')
        else:
            electrons_move_outward = False
            field_direction = -1
            logger.debug('Electrond move inward.')

        while pos[1] > ythresh and length < trajectory_length_cutoff and (pos[1] <= x_to_y_on_stamp_bottom_line(pos[0])): #(boundary.contains(Point(pos[0], pos[1])))
            if length > sample_dimension/number_of_charge_points and pos[1] > x_to_y_on_stamp_bottom_line(pos[0])*0.990:
                break
            # smallest of the two distances from current location to the edges
            smallest_separation = min([abs(pos[1]), abs(x_to_y_on_stamp_bottom_line(pos[0]) - pos[1])])
            if length > 0.005:
                dt = 0.005 + 0.05*smallest_separation
            if dt > 0.05:
                dt = 0.05
            try:
                E_here = -1*field_direction*field_at_points(pos)[0]
            except ValueError as err:
                print(err)
                return False, -1, 1
            step = E_here/np.linalg.norm(E_here) * dt
            length += dt
            pos += step
            trajectory.append(np.copy(pos))
            # test whether it's within the boundary still
            if Emax < np.linalg.norm(E_here):
                Emax = np.linalg.norm(E_here)
        trajectory = np.array(trajectory)
        if length>=trajectory_length_cutoff:
            trajfilename = '{0:.4f}_{1}.npy'.format(stamp_coverage,
                                                    np.random.randint(10000000))
            np.save(target_folder + 'datasets/long_trajectories/'+trajfilename,
                    trajectory)
            logger.debug('Saved long trajectory to {0}'.format(trajfilename))
        spark_is_up = (pos[1] > x_to_y_on_stamp_bottom_line(pos[0]))
        if spark_is_up:
            logger.debug('SPARK IS UP')
            end_of_trajectory_x = trajectory[-1, 0]
            x_step = sample_dimension/charge_points_x.shape[0]
            # spark_to_index = int(np.round((end_of_trajectory_x - (-1*sample_dimension/2))/x_step))
            spark_to_index = find_nearest(charge_points_x, end_of_trajectory_x)
        else:
            spark_to_index = -1
        # print('gap = {0:.5f}, Emax = {1:09.0f}, sparkup={2}'.format(length, Emax))
        breakdown_here = breakdown_field_correction * breakdown_field_at_gap(length*1e-3)
        linewidth = 1
        # breakdown_here = 3e6
        for plot_ax in [ax_illustr, ax_illustr_zoom]: # TODO: Make thick sparks in ax_illustr
            if len(trajectory)>0:
                if Emax > breakdown_here:
                    plot_ax.plot(trajectory[:,0], trajectory[:,1], color = 'darkorchid', linewidth=3*linewidth, alpha = 0.7)
                    traj_is_sparking = True
                    # np.save(folder_for_datasets + 'trajectories/R_{0:.8f}'.format(starting_point[0]), trajectory)
                else:
                    if electrons_move_outward:
                        plot_ax.plot(trajectory[:, 0], trajectory[:, 1], color='grey', linewidth=linewidth, alpha = 0.7)
                        # np.save(folder_for_datasets + 'trajectories/B_{0:.8f}'.format(starting_point[0]), trajectory)
                    else:
                        plot_ax.plot(trajectory[:, 0], trajectory[:, 1], color='grey', linewidth=linewidth, alpha = 0.7)
                        # np.save(folder_for_datasets + 'trajectories/G_{0:.8f}'.format(starting_point[0]), trajectory)
                    pass
        if traj_is_sparking:
            color_here = 'darkorchid'
            # np.save(folder_for_datasets + 'pasch/R_{0:.8f}'.format(starting_point[0]),
            #         Emax)
        else:
            color_here = 'grey'
            # np.save(folder_for_datasets + 'pasch/B_{0:.8f}'.format(starting_point[0]),
            #         Emax)
        axpasch.scatter(length / 1e-3, Emax, color=color_here,
                        alpha=0.5, s=8)
        return traj_is_sparking, spark_to_index, field_direction

    #make folder for saving datasets
    folder_for_datasets = target_folder + \
                          'datasets/iterations/iter_{0:06d}_coverage_{1:.4f}/'.format(iter_id, stamp_coverage)
    # os.mkdir(folder_for_datasets)
    # os.mkdir(folder_for_datasets + 'trajectories/')
    # os.mkdir(folder_for_datasets + 'pasch/')

    # plot paschen curve
    axpasch.clear()
    for plot_ax in [ax_illustr, ax_illustr_zoom]:
        plot_ax.clear()
        plot_ax.set_aspect('equal', 'box')
        plot_ax.set_axis_off()
    # axpasch.set_ylim([0, 6e8])
    axpasch.set_xscale('log')
    axpasch.set_yscale('log')

    wsds = (6e-3 + 7e-3)/2
    angle = 10*(np.pi/180)
    sample_dimension = 50
    delam_point_x = sample_dimension*(0.5-(1-stamp_coverage))
    # starting_list = np.linspace(delam_point_x+5e-2, delam_point_x + sample_dimension*(1-stamp_coverage)*np.cos(angle), 50)
    # Select bins that are between delam.point and the end of stamp

    # _mask = np.logical_and(charge_points_x > delam_point_x, # +5e-2,
    #                        charge_points_x < delam_point_x + sample_dimension*(1-stamp_coverage)*np.cos(angle))
    _mask = charge_points_x > delam_point_x
    starting_list = np.copy(charge_points_x[_mask])
    # starting_list = np.copy(charge_points_x[charge_points_x > delam_point_x])
    old_discharges_with_mask = np.copy(old_discharge_locations[_mask])
    indices_in_this_section = np.arange(charge_points_x.shape[0])[_mask]
    # discharge_list = []
    discharge_locations = np.zeros_like(charge_points_x, dtype=int)
    for i, starting_x in enumerate(starting_list):
        starting_point = np.array((starting_x, 0.995*(starting_x-delam_point_x)*np.tan(angle))) # *0.95
        discharge_hysteresis_factor_here = 1
        if old_discharges_with_mask[i] != 0:
            # TODO: do not hardcode global hysteresis factor here (now taken from [G. Whitesides, JACS 2007])
            discharge_hysteresis_factor_here = 3/8
        discharge_is_on, spark_to_index, field_direction = process_trajectory(starting_point,
                                                                              folder_for_datasets=folder_for_datasets,
                                             breakdown_field_correction = discharge_hysteresis_factor_here)
        if spark_to_index == indices_in_this_section[i]: # this means that this charge element shorts to itself
            logger.debug('Shorting to itself: local index {0}'.format(i))
            discharge_is_on = False
        # discharge_list.append(discharge_is_on)
        if discharge_is_on:
            discharge_locations[indices_in_this_section[i]] += 1*field_direction
            if spark_to_index >= 0 and spark_to_index < discharge_locations.shape[0]:
                discharge_locations[spark_to_index] -= 1*field_direction

    # discharge_locations[_mask] = np.array(discharge_list)
    # This function will return the charge point ids where breakdown is happening
    # plt.tight_layout()
    # fig.savefig(target_folder + 'frames/{0:08d}.png'.format(iter_id), dpi=800)
    # plt.show()
    # fig.clf()
    # del fig
    return discharge_locations

def boolarr_to_str(arr, decimateby=1):
    s = ''
    for x in arr[::decimateby]:
        if x:
            if x > 0:
                s += '8'
            else:
                s += '^'
        else:
            s += '_'
    return s

# discharge_locations = find_discharges(charge_density, stamp_coverage)
target_folder = 'results/air_pcd9_5cm_500pts_hyst3o8_rampend6_cstep_0p25_' + datetime.now().strftime("%m_%d_%Y-%H_%M_%S") + '/'
max_iterations_per_spark = 140
charge_flow_per_iteration = 0.25 # in nC/cm^2
stamp_coverage_step = 0.01*10/sample_dimension#/5
os.mkdir(target_folder)
os.mkdir(target_folder + 'frames/')
os.mkdir(target_folder + 'frames_pickle/')
os.mkdir(target_folder + 'cdensities/')
os.mkdir(target_folder + 'datasets/')
os.mkdir(target_folder + 'datasets/long_trajectories/')
os.mkdir(target_folder + 'datasets/iterations/')
os.mkdir(target_folder + 'datasets/charge_densities/')

# create logger
logger = logging.getLogger('main_application')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(target_folder + 'logging.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.debug('Created folders.')

# TODO: Change this to 0.98
stamp_coverage = 0.99
stamp_coverage_list = []

# here I make very negative charge density at some points just for testing purposes
# TODO: Remove this line, it is here for testing
# charge_density[-10:-3] = -5*prederjaguin_charge_density*(1e-9)/(1e-2)**2 # in C/m^2
charge_density[-3:] = charge_density[-3]*np.linspace(1, 0, 3) # make the very end of the stamp noncharged
# stamp_coverage = 0.05
# discharge_locations = find_discharges(charge_density, stamp_coverage, iter_id, target_folder)
# plt.show()
f_chargedensity, ax_cdensity = plt.subplots()
# ax_cdensity.set_aspect(aspect="auto")
# ax_cdensity.set_yticklabels([])
# ax_cdensity.set_xticklabels([])
divider = make_axes_locatable(ax_cdensity)
cax = divider.append_axes('right', size='5%', pad=0.05)


# fig, (ax_illustr, axpasch) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [3, 1]})
fig = plt.figure(constrained_layout=True, figsize=figsize)
gs = fig.add_gridspec(3, 3)
ax_illustr = fig.add_subplot(gs[0, :])
ax_illustr_zoom = fig.add_subplot(gs[1, :-1])
axpasch = fig.add_subplot(gs[1, 2])
ax_sigma = fig.add_subplot(gs[2, :])

ax_illustr.set_title('Stamp coverage: {0:.03f}'.format(stamp_coverage))
axpasch.set_xlabel('Trajectory length, $\mu$m')
axpasch.set_ylabel('Field strength, V/m')
# plt.quiver(xi,yi,Ex,Ey)
# ax.streamplot(xi,yi,Ex,Ey,density=3)
interp_box = {'x': [-6, 10], 'y': [1e-6, 3]}
ax_illustr.axis([interp_box['x'][0] * 1.1, interp_box['x'][1] * 1.1, 0, interp_box['y'][1]])
ax_illustr.set_aspect('equal', 'box')
ax_illustr.set_axis_off()

ax_illustr_zoom.set_aspect('equal', 'box')
ax_illustr_zoom.set_axis_off()
# axpasch.set_ylim([0, 6e8])
axpasch.set_xscale('log')
axpasch.set_yscale('log')
# boundary = air_boundary(stamp_coverage, ax)

# figpasch, axpasch = plt.subplots()
gaps_ = np.logspace(-6, -2, 30)
gaps_for_pasch = gaps_ / 1e-6
breakdowns_for_pasch_plot = breakdown_field_at_gap(gaps_)
axpasch.plot(gaps_for_pasch, breakdowns_for_pasch_plot, color='r')
axpasch.plot(gaps_for_pasch, breakdowns_for_pasch_plot*3/8, '--', color='r')

ax_sigma.plot(charge_points_x, charge_density)
ax_sigma.axhline(y=0)

plt.show(block=False)

# for_imshow = charge_density / 1e-5
# # for_imshow[charge_points_x < delam_point_x] = np.nan
# # ax_cdensity.clear()
# # ax_cdensity.set_yticklabels([])
# # ax_cdensity.set_xticklabels([])
# im = ax_cdensity.imshow(np.vstack((for_imshow, for_imshow)), cmap='seismic',
#                         vmax=prederjaguin_charge_density, vmin=-1 * prederjaguin_charge_density)
# ax_cdensity.set_aspect(aspect="auto")
# # if no_colorbar_yet:
# f_chargedensity.colorbar(im, cax=cax, orientation='vertical')
# # no_colorbar_yet = False
# # plt.colorbar(im)
# plt.draw()
# plt.pause(0.3)
# plt.show()

no_colorbar_yet = True
charge_points_x = get_charge_points_x(stamp_coverage, angle, npoints=number_of_charge_points)
discharge_locations = np.zeros_like(charge_points_x, dtype=bool)
discharge_lasted_for = 0
while stamp_coverage > 0.02:
    def meantime_procedures():
        # plt.draw()
        # plt.pause(0.1)
        t1 = time.time()
        fig.savefig(target_folder + 'frames/{0:08d}.png'.format(iter_id), dpi=300)
        # pickle.dump(fig, open(target_folder + 'frames_pickle/{0:08d}.pkl'.format(iter_id), 'wb'))
        logger.debug('Time spend saving fig: {0:.2f} sec.'.format(time.time() - t1))
        # plt.draw()
        # plt.pause(0.2)
    discharge_locations = find_discharges(charge_density, stamp_coverage, iter_id, target_folder, axpasch, ax_illustr,
                                          ax_illustr_zoom,
                                          old_discharge_locations = np.copy(discharge_locations),
                                          to_run_in_meantime = meantime_procedures)
    axpasch.plot(gaps_for_pasch, breakdowns_for_pasch_plot, color='C1')
    axpasch.plot(gaps_for_pasch, breakdowns_for_pasch_plot * 3 / 8, '--', color='C1', alpha=0.5)
    axpasch.set_xlim(8, 15000)
    axpasch.set_ylim(0.3*0.7e6, 5e7)
    axpasch.set_ylabel('Electric field (V/m)')
    axpasch.set_xlabel('Effective gap ($\mu$m)')

    ax_illustr.set_title('Progress: {0:.01f}% ({1:.02f} mm)'.format((1-stamp_coverage)*100, sample_dimension*(1-stamp_coverage)))
    iter_id += 1
    ax_sigma.clear()
    ax_sigma.set_ylabel('Charge density, $C/m^2$')
    ax_sigma.yaxis.set_major_formatter(mticker.FuncFormatter(g_scinotation))
    ax_sigma.set_xlabel('Coordinate along the sample, mm')
    ax_sigma.set_xlim([-sample_dimension/2, sample_dimension/2])
    ax_sigma.set_ylim([-1.5*prederjaguin_charge_density*(1e-9)/(1e-2)**2,
                       1.5*prederjaguin_charge_density*(1e-9)/(1e-2)**2])
    delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
    ax_sigma.fill_between(charge_points_x[charge_points_x > delam_point_x], 0,
                          charge_density[charge_points_x > delam_point_x],
                          where=charge_density[charge_points_x > delam_point_x] > 0,
                          facecolor='red', alpha=0.5, interpolate=True)
    ax_sigma.fill_between(charge_points_x[charge_points_x > delam_point_x], 0,
                          charge_density[charge_points_x > delam_point_x],
                          where=charge_density[charge_points_x > delam_point_x] < 0,
                          facecolor='blue', alpha=0.5, interpolate=True)
    ax_sigma.axhline(y=0, color='grey')
    np.save(target_folder + 'datasets/charge_densities/iter_{0:06d}_coverage_{1:.4f}'.format(iter_id,
                                                                                         stamp_coverage),
            charge_density)
    if np.any(discharge_locations) and discharge_lasted_for < max_iterations_per_spark:
        discharge_lasted_for += 1
        # decrease charge in these points
        logger.debug(boolarr_to_str(discharge_locations, decimateby=5))
        charge_density -= 1e-5*charge_flow_per_iteration*discharge_locations
        # plt.draw()
        # plt.pause(0.2)
    else:
        stamp_coverage_list.append(stamp_coverage)
        np.save(target_folder + 'datasets/coverage_{0:.3f}'.format(stamp_coverage), charge_density)
        delam_point_x = sample_dimension * (0.5 - (1 - stamp_coverage))
        # for_imshow = charge_density / 1e-5
        # for_imshow[charge_points_x<delam_point_x] = np.nan
        # ax_cdensity.clear()
        # im = ax_cdensity.imshow(np.vstack((for_imshow, for_imshow)), cmap='seismic',
        #            vmax=prederjaguin_charge_density, vmin=-1*prederjaguin_charge_density)
        # ax_cdensity.set_aspect(aspect="auto")
        # ax_cdensity.set_yticklabels([])
        # ax_cdensity.set_xticklabels([])
        # if no_colorbar_yet:
        # f_chargedensity.colorbar(im, cax=cax, orientation='vertical')
            # no_colorbar_yet = False
        # plt.draw()
        # plt.pause(0.2)
        t0 = time.time()
        # f_chargedensity.savefig(target_folder + 'cdensities/coverage_{0:.3f}.png'.format(stamp_coverage), dpi=200)
        logger.debug('Time spent saving charge distribution figure: {0:.2f} s'.format(time.time() - t0))

        stamp_coverage -= stamp_coverage_step
        charge_points_x = get_charge_points_x(stamp_coverage, angle, npoints=number_of_charge_points)
        discharge_lasted_for = 0
        discharge_locations = np.zeros_like(charge_points_x, dtype=bool)
        logger.debug('NEW STAMP COVERAGE = {0}'.format(stamp_coverage))

# TODO: Measure time for different operations and optimize. Profiling, in other words.
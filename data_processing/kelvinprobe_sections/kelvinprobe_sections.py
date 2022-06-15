import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import pylab
from skimage.transform import resize
import matplotlib.colors as colors
import scipy.linalg
import json
import pickle
from matplotlib.ticker import LogLocator

epsilon = 4 # dielectric constant
thickness = 0.55e-6 #m
area = (1e-2)**2 # m^2
capacitance = epsilon*(8.85e-12)*area/thickness

def fit_plane_to_points(pot, data):
    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(pot.shape[0]), np.arange(pot.shape[1]))
    XX = X.flatten()
    YY = Y.flatten()

    order = 1  # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
    return Z

def take_KP_section(target, filename_for_saving, processed = True, do_plotting=False, turn_angle = 0, stripe_loc=[10,20],
                           ylim = 3400, bkg=False,
                 bkg_pos='left', maxval=False, window_size_x = 40, window_size_y = 6):
    if processed:
        data = np.load('{0}/processed/data/data.npy'.format(target))
        position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    else:
        data = np.load('{0}/data/data.npy'.format(target))
        position_griddata = np.load('{0}/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    pot = -1*data[1]/1000
    maxx = np.max(X)
    minx = np.min(X)
    maxy = np.max(Y)
    miny = np.min(Y)
    height_in_mm = maxy - miny
    width_in_mm = maxx - minx
    # bkg = np.mean(pot[:1000, :35])
    # pot = pot - bkg

    print('Final image size is {0}'.format(pot.shape))
    if not bkg:
        if bkg_pos == 'left':
            bkg = np.mean(pot[:, 10:20])
        elif bkg_pos == 'right':
            bkg = np.mean(pot[:, -20:-10])
        else:
            pot_masked = np.copy(pot)
            bkg_points = []
            for a_point in bkg_pos:
                bkg_at_this_point = np.mean(pot[a_point[0]-window_size_x:a_point[0]+window_size_x,
                                                a_point[1]-window_size_y:a_point[1]+window_size_y])
                pot_masked[ a_point[0] - window_size_x:a_point[0] + window_size_x,
                            a_point[1] - window_size_y:a_point[1] + window_size_y] =30
                bkg_points.append([a_point[0], a_point[1], bkg_at_this_point])
            bkg_points = np.array(bkg_points)
            bkg = fit_plane_to_points(pot, bkg_points)

        print('Background is {0}'.format(bkg))
    pot = pot - bkg.T

    height_in_pixels = X.shape[0]
    width_in_pixels = X.shape[1]
    target_pixels_per_mm = height_in_pixels/height_in_mm
    target_width_in_pixels = int(round(target_pixels_per_mm * width_in_mm))
    resized_map = resize(pot, output_shape=(height_in_pixels, target_width_in_pixels), anti_aliasing=False,
                         order=0)
    # bkg = np.mean(pot[:,30:67])
    map_of_charge_density = capacitance*resized_map/1e-9 # now it is in nC/cm^2

    if turn_angle:
        map_of_charge_density = ndimage.rotate(map_of_charge_density, angle=turn_angle)

    fig_chargedensity = plt.figure(333)
    ims = plt.imshow(map_of_charge_density, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval),
               cmap = 'seismic')
    plt.axis('scaled')
    plt.colorbar()
    fig_chargedensity.savefig(target + '\\graphs\\charge_density_bkg_corrected.png', dpi=300)
    plt.show()

    stripe = np.mean(map_of_charge_density[:ylim,stripe_loc[0]:stripe_loc[1]], axis=1)
    # stripe = stripe - np.linspace(0, -1.75, stripe.shape[0])
    stripe_x = Y[:ylim, 0]
    f2 = plt.figure(2)
    plt.plot(stripe_x, stripe)


    f3, ax_sigma = plt.subplots(1, figsize=(10,2.75))
    ax_sigma.fill_between(stripe_x, 0,
                          stripe,
                          where=stripe > 0, facecolor='red', alpha=0.5, interpolate=True)
    ax_sigma.fill_between(stripe_x, 0,
                          stripe,
                          where=stripe < 0, facecolor='blue', alpha=0.5, interpolate=True)
    ax_sigma.axhline(y=0, color='grey')

    f3.savefig(target + '\\graphs\\charge_density_section.png', dpi=300)
    plt.show()


# plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
#                    bkg_pos='left',
#                    maxval=2.4)


# base_folder = 'experimental_data\\improved_kp_kelvinprobe\\'
# filename_list = [
#     '20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
#     '20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
#     '20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
#     ]
# processing_log = []
# for i,file in enumerate(filename_list):
#     filename_for_saving = '{0:03d}'.format(i)
#     take_2D_power_spectrum(base_folder + file, filename_for_saving=filename_for_saving)
#     processing_log.append([base_folder + file, filename_for_saving])
#
# with open('kelvinprobe/processing_log.txt', 'w+') as fout:
#     json.dump(processing_log, fout)
# pickle.dump(processing_log, open('kelvinprobe/processing_log.pickle', 'wb'))


# take_2D_power_spectrum('experimental_data\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
#                    filename_for_saving='004',
#                        stripe_loc=[85, 118],
#                        do_plotting=True)

# take_KP_section('experimental_data\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
#                    filename_for_saving='004',
#                        turn_angle=6,
#                        stripe_loc=[568, 791],
#                        do_plotting=True)

# take_KP_section('experimental_data\\improved_kp_kelvinprobe\\20191030_5cm_3in_50RH_ambRH38_eq30min_newPDMS5to1_PMMAtol_uniformspeeds_0p5_C01_copy',
#                    filename_for_saving='004',
#                        turn_angle=6,
#                        stripe_loc=[649, 651],
#                        do_plotting=True,
#                 bkg_pos=[[256, 63], [1952, 193], [3235, 46]],
#                 ylim = 3500)

# take_KP_section('experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',
#                    filename_for_saving='004',
#                        turn_angle=0,
#                        stripe_loc=[634, 636],
#                        do_plotting=True,
#                 bkg_pos=[[213, 85], [3383, 107], [3341, 191]],
#                 ylim = 3500)

# take_KP_section('experimental_data\\improved_kp_kelvinprobe\\'
#                 '20200123__Argonauts_5cm_3in_64RH_2hydrometer_68RH_ambRH25_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p4_E01',
#                    filename_for_saving='004',
#                        turn_angle=6,
#                        stripe_loc=[930, 1050],#[989, 990],
#                        do_plotting=True,
#                 bkg_pos=[[3541, 178], [3454, 88], [100, 186]],
#                 ylim = 3800)

# take_KP_section('experimental_data\\improved_kp_kelvinprobe\\20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
#                    filename_for_saving='004',
#                        turn_angle=6,
#                        stripe_loc=[649, 651],
#                        do_plotting=True,
#                 bkg_pos=[[256, 63], [1952, 193], [3235, 46]],
#                 ylim = 3500)

take_KP_section('E:\\PDMS-PMMA_delamination_experiments\\kelvin_probe\\'
                '20191108_5cm_3in_31RH_ambRH32_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_E01',
                   filename_for_saving='08102',
                       turn_angle=0,
                       stripe_loc=[438, 470],
                       do_plotting=True,
                        maxval=30,
                bkg_pos=[[160, 62], [3001, 62], [3775, 144], [3086, 222], [2609, 216], [86, 211], [54, 68]],
                ylim = 3500)
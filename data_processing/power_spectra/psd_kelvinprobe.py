import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab
from skimage.transform import resize
from scipy import signal
import json
import pickle
from matplotlib.ticker import LogLocator
from scipy import ndimage
import scipy.linalg

# im0 = imageio.imread('spdtest.png')
# im = im0[:,:]

# im0 = imageio.imread('spdtest.jpg')
# im = np.sum(im0, axis=2)

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

def take_2D_power_spectrum(target, filename_for_saving, processed = True, do_plotting=False,
                           vcut=False, cut=False, turn_angle=0,
                           bkg=False,
                           bkg_pos='left',
                           window_size_x = 40,
                           window_size_y = 6):
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
    target_width_in_pixels = int(round(target_pixels_per_mm*width_in_mm))
    resized_map = resize(pot, output_shape=(height_in_pixels, target_width_in_pixels), anti_aliasing=False,
                         order=0)
    resized_map_of_charge_density = capacitance*resized_map/1e-9 # now it is in nC/cm^2

    if turn_angle:
        resized_map_of_charge_density = ndimage.rotate(resized_map_of_charge_density, angle=turn_angle)

    if do_plotting:
        plt.imshow(resized_map_of_charge_density)

    # if window:
    #     resized_map_of_charge_density_for_spectrum =

    hanning = True
    freqG, psdG = psds.power_spectrum(resized_map_of_charge_density, oned=True, hanning=hanning)
    # freqGB, psdGB = psds.power_spectrum(resized_map_of_charge_density[:2840,272:856], oned=True, hanning=hanning)
    # freqGC, psdGC = psds.power_spectrum(resized_map_of_charge_density[:, 929:], oned=True, hanning=hanning)
    area = resized_map_of_charge_density.shape[0]*resized_map_of_charge_density.shape[1]/(target_pixels_per_mm)**2
    data_for_export = np.vstack((freqG*(target_pixels_per_mm/np.sqrt(2)), psdG/area))
    np.save('kelvinprobe/'+filename_for_saving, data_for_export)
    if do_plotting:
        f = plt.figure(2, figsize=(10,10))
        # pylab.clf()
        ax = f.add_subplot(111)
        plt.yscale('log')
        plt.xscale('log', subsx=[-1, 0, 1])
        # ax.yaxis.set_major_locator(LogLocator(base=10))
        # ax.xaxis.set_major_locator(LogLocator(base=10))
        # ax.xaxis.set_minor_locator('none')
        ax.plot(freqG*(target_pixels_per_mm/np.sqrt(2)), psdG/area, label='Power spectrum')
        # ax.plot(freqGB * target_pixels_per_mm, psdGB, label='Power spectrum, stamp only')
        # ax.plot(freqGC * target_pixels_per_mm, psdGC, label='Power spectrum, noise vertical')
        # pylab.legend(loc='best')
        # pylab.axis([7,400,1e-7,1])
        plt.xlabel("Spatial frequency ($mm^{-1}$)")
        plt.ylabel("Normalized Power")
        # ax.yaxis.set_major_locator(LogLocator(base=100))
        ax.grid(True, which="both", ls="-")
        plt.legend()

    # get psd of vertical sections
    if not cut:
        cut = [0, -1]
    if vcut:
        resized_map_of_charge_density = resized_map_of_charge_density[vcut[0]:vcut[1]]
    spectra = []
    fs = target_pixels_per_mm
    for i in list(range(resized_map_of_charge_density.shape[1]))[cut[0]:cut[1]]:
        pcd = resized_map_of_charge_density[:,i]
        # if i == 500:
        #     fz = plt.figure(10)
        #     plt.plot(pcd)
        #     plt.show()
        f, Pxx_den = signal.welch(pcd, fs, nfft=len(pcd), detrend=False)
        spectra.append(np.copy(Pxx_den))
    spectra = np.array(spectra)
    Pxx_den = np.mean(spectra, axis=0)
    plt.loglog(f, Pxx_den, '-', alpha=0.6)
    data_for_export = np.vstack((f, Pxx_den))
    np.save('kelvinprobe/1d/'+filename_for_saving, data_for_export)
    plt.show()



# plot_bipolar(_uniformspeeds_slowstart_dark_0p4_A01',
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

#
# take_2D_power_spectrum('experimental_data\\improved_kp_kelvinprobe\\20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
#                    filename_for_saving='004_test',
#                        do_plotting=True)

take_2D_power_spectrum('experimental_data\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
                   filename_for_saving='20191123_70RH_B01',
                       do_plotting=True,
                       cut=[350, 969],
                       vcut=[225, 3554],
bkg_pos=[[52, 166], [2450, 7], [3288, 23], [1125, 24], [200, 42], [499, 185], [2715, 136], [1782, 160]],
                       turn_angle=6)

# 'experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy'

# take_2D_power_spectrum('experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',
#                    filename_for_saving='20191015_62RH_B01',
#                        do_plotting=True,
#                        cut=[501, 1063],
#                        vcut=[0, 3468],
#                        turn_angle=0,
#                        bkg_pos=[[213, 85], [3383, 107], [3341, 191]])

# take_2D_power_spectrum('experimental_data\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
#                    filename_for_saving='005_test',
#                        do_plotting=True,
#                        cut=[380, 869])

#
plt.show()
# im0 = imageio.imread('D:\\Docs\\Science\\UNIST\\Projects\\Vitektrification\\paper_at_all_scales\\pics\\input\\for_power_spectra\\output1.tif')
# im = im0[:,:]
#
# plt.imshow(im0)
# plt.show()
#
# # xx = psds.PSD2(im)
# # plt.imshow(np.log(xx))
# # plt.colorbar()
# # plt.show()
#
# freqG,psdG = psds.power_spectrum(im, oned=True)
# print(1)
# pylab.figure(2)
# pylab.clf()
# pylab.loglog(freqG,psdG,label='Power spectrum')
# pylab.legend(loc='best')
# # pylab.axis([7,400,1e-7,1])
# pylab.xlabel("Spatial frequency ($pixels^{-1}$)")
# pylab.ylabel("Normalized Power")
# pylab.grid(True,which="both",ls="-")
# # pylab.savefig("example_psd_scale.png")


# pylab.savefig("example_psd_scale.png")

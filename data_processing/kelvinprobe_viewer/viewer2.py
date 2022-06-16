# -*- coding: utf-8 -*-
"""
Plotter of Kelvin probe scans

Created on Thu May 10 10:25:24 2018

@author: Yaroslav I. Sobolev
"""

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
#from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#import matplotlib.font_manager as fm
#fontprops = fm.FontProperties(size=18)
from matplotlib import lines
import os
import glob
import scipy.linalg
import skimage
import pickle
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit

#
#font = {'size' : 18}
#matplotlib.rc('font', **font)
#matplotlib.rc('font', **{'family':'serif','serif':['Palatino']})

epsilon = 4 # dielectric constant
thickness = 1e-6 #m
area = (1e-2)**2 # m^2
# capacitance = epsilon*(8.85e-12)*area/thickness

def write_xyz_file(X,Y,Z, save_to_filename, units_str = 'V'):
    xyz_file = open(save_to_filename, 'w+')
    xyz_file.write('# Channel: Detail 2\n')
    xyz_file.write('# Lateral units: mm\n')
    xyz_file.write('# Value units: {0}\n'.format(units_str))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xyz_file.write("{0:.7e}\t{1:.7e}\t{2:.7e}\n".format(
                           X[i,j], Y[i,j], Z[i,j]))
    xyz_file.close()

def extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence, dt = 0):
    time_sequence = np.array(time_sequence)
    data = np.genfromtxt(filename, skip_header=1, skip_footer=21, delimiter=',')
    nrows, ncols = data.shape
    new_data = np.zeros(shape=(len(time_sequence), ncols), dtype = np.float64)
    for colid in range(ncols):
        new_col = np.interp(time_sequence+dt, data[:,-1], data[:,colid])
        new_data[:,colid] = new_col
    return new_data

def extract_from_datfile(filename):
    data = np.genfromtxt(filename, skip_header=1, skip_footer=21, delimiter=',')
    return data
    
def interpolate_at_timepoints(data, time_sequence, dt = 0):
    nrows, ncols = data.shape
    new_data = np.zeros(shape=(len(time_sequence), ncols), dtype = np.float64)
    for colid in range(ncols):
        new_col = np.interp(time_sequence+dt, data[:,-1], data[:,colid])
        new_data[:,colid] = new_col
    return new_data    

def correct_shifts_for_target(target, force_zero_shifts = False):
    frags = list(os.walk(target + '/fragments'))[0][1]
    main_folder = target + '/processed'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(main_folder + '/data'):
        os.makedirs(main_folder + '/data')
    if not os.path.exists(main_folder + '/graphs'):
        os.makedirs(main_folder + '/graphs')
    fragment_minshifts = []
    for nfrag, f in enumerate(frags):
        print('Processing frag {0}'.format(f))
        folder_for_fragment = '{0}/fragments/{1}'.format(target, f)
    #    data = np.load('{0}/data/data.npy'.format(target))
        position_griddata = np.load('{0}/position_griddata.npy'.format(folder_for_fragment))
        time_sequence = np.load('{0}/time_sequence.npy'.format(folder_for_fragment))
        datfile = [file for file in glob.glob("{0}/*.dat".format(folder_for_fragment))][0]
    
        data = extract_from_datfile(datfile)
        mismatches = []
        dts = np.linspace(-2,2,150)
        for ddts_k, dt in enumerate(dts):
            new_data = interpolate_at_timepoints(data, time_sequence, dt=dt)
            WF = np.reshape(new_data[:,1], (position_griddata[0].shape[1], position_griddata[0].shape[0])).T
            diffs = np.square(np.gradient(WF, axis=1, edge_order = 2))
            avg_diff = np.mean(diffs[3:-2])
            mismatches.append(avg_diff)
        plt.plot(mismatches)
        # plt.show()
        min_index = np.argmin(mismatches)
        if min_index == 0 or min_index == len(mismatches)-1 or \
            ((max(mismatches) - min(mismatches)) < 0.001) or \
            force_zero_shifts:
            minshift = 0
        else:
            minshift = dts[np.argmin(mismatches)]
        fragment_minshifts.append(minshift)
        print('Found a proper shift: {0}'.format(minshift))
        data = interpolate_at_timepoints(data, time_sequence, dt=minshift)
    #    data = extract_from_datfile_and_interpolate_at_timepoints(datfile, time_sequence, dt=0)
        X,Y = position_griddata
    #    position_sequence, data, time_sequence, position_griddata = \
    #            scan_small_fragment([x_begin, begin[1]], 
    #                                [x_end, end[1]], 
    #                                x_piece_size, 
    #                                N_pts_y, 
    #                                folder_for_fragment)
        data3d = []
        for k in range(data.shape[1]):    
            Y = np.reshape(data[:,k], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
            data3d.append(Y.T)
        data3d = np.array(data3d)
        if nfrag == 0:
            total_position_grid = position_griddata
            total_data = data3d
            total_time_sequence = np.array(time_sequence)
        else:
            total_position_grid = np.concatenate((total_position_grid, position_griddata), axis=2)
            total_data = np.concatenate((total_data, data3d), axis=2)
            total_time_sequence = np.concatenate((total_time_sequence, np.array(time_sequence)))
            
    np.save(main_folder + '\\data\\time_sequence.npy', np.array(total_time_sequence))
    np.save(main_folder + '\\data\\data.npy', total_data)
    np.save(main_folder + '\\data\\position_griddata.npy', total_position_grid)    

    np.save(main_folder + '\\data\\fragment_minshifts.npy', np.array(fragment_minshifts))
    f1 = plt.figure(1, figsize=(6,4))
#        ax = plt.subplot(211)
    plt.title('WF')
    #    Y = np.reshape(data[:,1], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
    plt.pcolor(total_position_grid[0], total_position_grid[1], total_data[1], cmap = 'viridis_r',
               norm=colors.Normalize(vmin=-7000, vmax=0))
    #plt.axes().set_aspect('equal')
    plt.axis('scaled')
    plt.colorbar()
#        ax = plt.subplot(212)
#        plt.title('Gradient')
#        plt.pcolor(total_position_grid[0], total_position_grid[1], total_data[5])
#        #    Y = np.reshape(data[:,5], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
#        plt.axis('scaled')
#        #plt.axes().set_aspect('equal')
#        plt.colorbar()
    f1.savefig(main_folder + '\\graphs\\WF_and_GD_processed.png', dpi = 1200)
    f1.clear()
    f1.clf()     

def plot_corrected_pics(target, bkg = 0, bkg_pos = 'left', cut_from_left = False):
    data = np.load('{0}/processed/data/data.npy'.format(target))
    position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    pot = -1*data[1]/1000
    plt.imshow(pot[:, 10:20])
    plt.show()
    if not bkg:
        if bkg_pos == 'left':
            bkg = np.mean(pot[:, 10:20])
        elif bkg_pos == 'right':
            bkg = np.mean(pot[:, -20:-10])
        print('Background is {0}'.format(bkg))
    pot = pot - bkg
    
    f1 = plt.figure(1)
    ax = f1.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    maxval = max(abs(pot[:, cut_from_left:].min()), abs(pot[:, cut_from_left:].max()))
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval),
               cmap = 'seismic')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    plt.colorbar()
    f1.savefig(target + '/processed/final_fig_bipolar.png', dpi=1200)
    plt.axis('off')
    f1.savefig(target + '/processed/final_fig_bipolar_noaxis.png', dpi=1200)
    f1.clf()
    
    f3 = plt.figure(3, figsize = (6,6))
    ax = f3.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=0, vmax=pot[:, cut_from_left:].max()),
               cmap = 'viridis')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    plt.colorbar()
    f3.savefig(target + '/processed/final_fig_negative_part.png', dpi=1200)
    plt.axis('off')
    f3.savefig(target + '/processed/final_fig_negative_part_noaxis.png', dpi=1200)
    f3.clf()
                                                                                                                                                                                                                                                                                                                                                    
#plot_corrected_pics('experiments/sample2_zoom_1', bkg = 0.669282680827785)

def fit_plane_to_points(pot, data, order):
    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(pot.shape[0]), np.arange(pot.shape[1]))
    XX = X.flatten()
    YY = Y.flatten()

    # order = 1  # 1: linear, 2: quadratic
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

def plot_bipolar(target, saveto = 'bipolar.png', processed = True, bkg=0,
                 bkg_pos='left', maxval=False, window_size_x = 40, window_size_y = 6, do_preview=False,
                 do_fig_clf=True, fitting_order=1, do_alignment=False,
                 window_for_max=[0, 3436, 40, 157], c_premax=3.6, nbins = 210,
                 right_edge_condition=1e-3,
                 scalebar_loc='lower right'):
    if processed:
        data = np.load('{0}/processed/data/data.npy'.format(target))
        position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    else:
        data = np.load('{0}/data/data.npy'.format(target))
        position_griddata = np.load('{0}/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    pot = -1*data[1]/1000

    # ys = []
    # for i in range(pot.shape[1]-1):
    #     diff = np.max(np.abs(pot[:,i+1] - pot[:,i]))
    #     ys.append(np.copy(diff))
    # plt.plot(ys)
    # plt.show()

    if do_alignment:
        diffthresh = 1.5
        shift_limit = 40
        for aligning_passes in range(5):
            print('Aligning pass {0} ===================='.format(aligning_passes))
            for i in range(pot.shape[1]-1):
                diff = np.max(np.abs(pot[:, i + 1] - pot[:, i]))
                if diff < diffthresh:
                    # print('unchanged column: {0}'.format(i))
                    continue
                shifted_diffs = []
                shift_list = np.arange(-shift_limit,shift_limit+1)
                for shift in shift_list:
                    rolled_potline = np.roll(pot[:,i+1], shift)
                    diff = np.max(np.abs(rolled_potline - pot[:,i]))
                    shifted_diffs.append(diff)
                best_shift = shift_list[np.argmin(shifted_diffs)]
                print('Best shift:{0}'.format(best_shift))
                left_shift = int(round(best_shift/2))
                right_shift = best_shift - left_shift
                # pot[:, i + 1] = np.roll(pot[:, i + 1], best_shift)
                pot[:, i + 1] = np.roll(pot[:, i + 1], left_shift)
                pot[:, i] = np.roll(pot[:, i], -1*right_shift)

    if do_preview:
        fig = plt.figure(0)
        ax0 = fig.add_subplot(111)
        ax0.imshow(pot)
        ax0.axis('auto')
        coords = []
        def onclick(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            print(
            'x = %d, y = %d' % (
                ix, iy))
            coords.append((ix, iy))
            np.savetxt(target + '/graphs/' + 'stamp_polygon_vertices.txt', np.array(coords))
            if event.button == 'down':
                fig.canvas.mpl_disconnect(cid)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
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
                            a_point[1] - window_size_y:a_point[1] + window_size_y] = 30
                bkg_points.append([a_point[0], a_point[1], bkg_at_this_point])
            bkg_points = np.array(bkg_points)
            bkg = fit_plane_to_points(pot, bkg_points, fitting_order)

        print('Background is {0}'.format(bkg))
    pot = pot - bkg.T
    f1 = plt.figure(1, figsize=(6,7))
    ax = f1.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    if not maxval:
        maxval = max(abs(pot.min()), abs(pot.max()))
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval),
               cmap = 'seismic')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    
    scalebar = ScaleBar(1 ,'mm', fixed_value=2,
                        location = scalebar_loc, font_properties={'size':0.03}, sep=-1,
                        frameon=False)
    ax.add_artist(scalebar)
    plt.axis('off')

    ## new clear axis overlay with 0-1 limits
    #ax3 = pyplot.axes([0,0,1,1], axisbg=(1,1,1,0))
    #
    #x,y = numpy.array([[0.05, 0.1, 0.9], [0.05, 0.5, 0.9]])
    #line = lines.Line2D(x, y, lw=5., color='green', alpha=1)
    #ax3.add_line(line)
    
    cbar = plt.colorbar(orientation='horizontal', fraction=0.03)
    cbar.set_label('$\phi, \mathrm{V}$')
    cbar.ax.xaxis.set_label_coords(0.95, -1.5)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    maxval_c = capacitance*maxval/1e-9
    ax2.set_xlim([-1*maxval_c,maxval_c])
    pos.x0 += 0.05
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    ax2.set_xlabel('$\sigma, \mathrm{nC/cm^2}$')
    ax2.xaxis.set_label_coords(0.95, 2.7)
    f1.savefig(target + '/' + saveto, dpi=1200)
    # plt.show()
    ax.pcolor(X,Y,pot_masked, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval))
    # pot_for_max = pot[window_for_max[0]:window_for_max[1], window_for_max[2]:window_for_max[3]]
    # location_of_max = np.unravel_index(pot_for_max.argmax(), pot_for_max.shape)
    # ax.scatter(X[location_of_max], location_of_max[0], c='red')
    # print(location_of_max)
    f1.savefig(target + '/' + 'bipolar_masked.png', dpi=1200)
    # plt.show()
    # f1.clf()

    # computing charge density averaged over the stamp
    stamp_vertices = np.loadtxt(target + '/graphs/' + 'stamp_polygon_vertices.txt')
    pot_mask = np.zeros_like(pot)
    poly_x, poly_y = skimage.draw.polygon(stamp_vertices[:, 1], stamp_vertices[:, 0], pot.shape)
    # poly_x, poly_y = skimage.draw.polygon2mask(pot.shape, stamp_polygon)
    pot_mask[poly_x, poly_y] = 1
    pixel_area = (X[0, 1] - X[0, 0])*(Y[1, 0] - Y[0, 0])  # in mm^2
    stamp_area = pixel_area*np.sum(pot_mask)
    print('Stamp area is {0} mm^2'.format(stamp_area))
    net_charge = np.sum(pot_mask*pot*capacitance/1e-9)*pixel_area/(1e2) # multiplied by area in mm^2
    average_charge_density = np.sum(pot_mask*pot*capacitance/1e-9)/np.sum(pot_mask)
    print('Net charge is {0} nC'.format(net_charge))
    print('Average charge density is {0} nC/cm^2'.format(average_charge_density))
    ax.pcolor(X, Y, pot_mask)
    plt.show()

    np.save(target + '/processed/charge_density_after_bkg_subtracted', pot*capacitance/1e-9)
    np.save(target + '/processed/stamp_mask', pot_mask)
    pot_for_max = pot[pot_mask==1]
    hist, bin_edges = np.histogram(pot_for_max*capacitance/1e-9, bins=nbins, density=True)
    horiz_thresh = c_premax
    horiz_thresh_in_bin_ids = np.argmax(bin_edges>horiz_thresh)
    hist_after_threshold = np.copy(hist)
    hist_after_threshold[:horiz_thresh_in_bin_ids] = -100
    histmax = np.argmax(hist_after_threshold)
    histmax_in_nC = (bin_edges[histmax] + bin_edges[histmax+1])/2
    hist_end = np.max(np.argwhere(hist > right_edge_condition))
    hist_end_in_nC = (bin_edges[hist_end] + bin_edges[hist_end+1])/2

    overall_histmax = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1])/2
    f3, ax3 = plt.subplots(figsize=(5,3))
    ax3.hist(pot_for_max*capacitance/1e-9, bins=nbins, density=True, color='grey', ec='grey')
    np.savetxt(target + '/' + 'max_density_metrics.txt', np.array([histmax_in_nC, hist_end_in_nC]),
               delimiter='\t', header='Histogram_max,_nC\tHistogram_right_edge,_nC')
    # plt.plot(bin_edges[:-1], hist)
    ax3.axvline(x=0, color='black', alpha=1, linewidth=3)
    ax3.axvline(x=histmax_in_nC, color='C1', alpha=0.5)
    ax3.axvline(x=hist_end_in_nC, color='C2', alpha=0.5)
    ax3.axvline(x=average_charge_density, linestyle='--', color='black', alpha=1)
    ax3.axvline(x=overall_histmax, color='blue', alpha=0.5)
    ax3.set_ylabel('Probability density, nC$^{-1}\cdot$cm$^{2}$')
    ax3.set_xlabel('Surface charge density, nC$\cdot$cm$^{-2}$')
    ax3.set_xlim(-10,10)
    # print('Max charge density={0}'.format(np.max(pot_for_max.flatten()*capacitance/1e-9)))
    plt.tight_layout()
    f3.savefig(target + '/' + 'histogram.png', dpi=300)
    plt.show()

def make_xyz_files(target):
    main_folder = target + '/xyz'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    data = np.load('{0}/data/data.npy'.format(target))
    position_griddata = np.load('{0}/data/position_griddata.npy'.format(target))
    X,Y = position_griddata    
    
    Z = -1*data[1]/1000
    write_xyz_file(X,Y,Z,main_folder+'/potential.xyz', units_str = 'V')
    Z = data[2]/1000
    write_xyz_file(X,Y,Z,main_folder+'/WFRA.xyz', units_str = 'V')
    Z = data[3]/1000
    write_xyz_file(X,Y,Z,main_folder+'/WFDel.xyz', units_str = 'V')
    Z = data[4]/1000
    write_xyz_file(X,Y,Z,main_folder+'/Std_WF.xyz', units_str = 'V')
    Z = data[5]/1000
    write_xyz_file(X,Y,Z,main_folder+'/GD.xyz', units_str = 'au')
    Z = data[6]/1000
    write_xyz_file(X,Y,Z,main_folder+'/Std_GD.xyz', units_str = 'au')
    Z = data[7]
    write_xyz_file(X,Y,Z,main_folder+'/Z_height.xyz', units_str = 'um')

def postprocess_folder(target, bkg_pos='left', force_zero_shifts = False, maxval=False):
    correct_shifts_for_target(target, force_zero_shifts = force_zero_shifts)
    make_xyz_files(target)
    plot_bipolar(target, bkg_pos=bkg_pos, maxval=maxval)
    plot_corrected_pics(target,bkg_pos=bkg_pos)

def get_moving_averaged_stats(target, saveto='movavg_stats.png', maxval=False,
                              was_precomputed=False,
                              violin_kernel_points=5,
                              plot_linear_fit=True
                              ):
    data = np.load('{0}/processed/data/data.npy'.format(target))
    position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    sigma_map = np.load(target + '/processed/charge_density_after_bkg_subtracted.npy')
    sigma_mask = np.load(target + '/processed/stamp_mask.npy')

    # f1 = plt.figure(1, figsize=(6, 7))
    # ax = f1.gca()
    # # plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    # #           cmap = 'viridis')
    # if not maxval:
    #     maxval = max(abs(sigma_map.min()), abs(sigma_map.max()))
    # plt.pcolor(X, Y, sigma_map, norm=colors.Normalize(vmin=-1 * maxval, vmax=maxval),
    #            cmap='seismic')
    # plt.axis('scaled')
    # plt.xlim([X.min(), X.max()])
    # plt.ylim([Y.min(), Y.max()])

    from skimage.morphology import rectangle
    from skimage.filters import rank
    from skimage.draw import rectangle as rectangle_draw

    step_size_in_x = (X[0, 1] - X[0, 0])
    step_size_in_y = (Y[1, 0] - Y[0, 0])
    list_of_rect_pix_widths = np.array([int(round(x)) for x in np.logspace(0.301, 2.07, 20)]) #np.array([5, 10, 20, 50, 100])
    if not was_precomputed:
        if not os.path.exists(target + '/processed/moving_averages/'):
            os.makedirs(target + '/processed/moving_averages/')
        for rect_pix_width in list_of_rect_pix_widths:
        # find the size of rectangle that looks like square in terms of real dimensions
            rect_pix_height = int(round(rect_pix_width*step_size_in_x/step_size_in_y))
            averaged_sigma = np.zeros_like(sigma_map)
            averaged_mask = np.zeros_like(sigma_mask)
            for start_x in range(averaged_sigma.shape[0]-rect_pix_height):
                print('window={0}, line={1}'.format(rect_pix_width, start_x))
                for start_y in range(averaged_sigma.shape[1] - rect_pix_width):
                    rr, cc = rectangle_draw(start=(start_x, start_y), extent=(rect_pix_height, rect_pix_width))
                    averaged_sigma[start_x, start_y] = np.mean(sigma_map[rr, cc])
                    averaged_mask[start_x, start_y] = np.mean(sigma_mask[rr, cc])
            np.save(target + '/processed/moving_averages/{0:03d}_sigma'.format(rect_pix_width), averaged_sigma)
            np.save(target + '/processed/moving_averages/{0:03d}_mask'.format(rect_pix_width), averaged_mask)

    f2, axarr = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    ax2 = axarr[1]
    ax1 = axarr[0]
    means = []
    stds = []
    abs_charge_densities_list = []
    charge_densities_list = []
    significant_pix_widths = []
    for rect_pix_width in list_of_rect_pix_widths:
        averaged_sigma = np.load(target + '/processed/moving_averages/{0:03d}_sigma.npy'.format(rect_pix_width))
        averaged_mask = np.load(target + '/processed/moving_averages/{0:03d}_mask.npy'.format(rect_pix_width))
        abs_sigmas = np.abs(averaged_sigma[averaged_mask == 1])
        if abs_sigmas.shape[0] == 0:
            continue
        abs_charge_densities_list.append(abs_sigmas)#*(step_size_in_x*rect_pix_width)**2/100)
        charge_densities_list.append(averaged_sigma[averaged_mask == 1])
        print('rect_width={0}, available_points={1}'.format(rect_pix_width, abs_sigmas.shape))
        means.append(np.mean(abs_sigmas))
        stds.append(np.std(abs_sigmas))
        significant_pix_widths.append(rect_pix_width)
    list_of_rect_pix_widths = np.array(significant_pix_widths)
    means = np.array(means)
    stds = np.array(stds)
    areas = (step_size_in_x*list_of_rect_pix_widths)**2/100
    xs = step_size_in_x*list_of_rect_pix_widths
    ys = areas*means

    parts = ax1.violinplot(
        charge_densities_list, xs,
        points=violin_kernel_points, showmeans=False, showmedians=False, widths=0.15*xs,
        showextrema=False)
    ax1.axhline(y=0, color='darkorchid', linestyle='--')

    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    ax1.set_ylabel('Window-averaged charge\ndensity ($\langle \sigma \\rangle _{W}$), nC cm$^{-2}$')

    parts = ax2.violinplot(
        [np.log10(x) for x in abs_charge_densities_list], xs,
        points=2*violin_kernel_points, showmeans=False, showmedians=False, widths=0.15*xs,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    ax2.scatter(xs, [np.log10(x) for x in means], color='C3')
    ax2.set_ylim(-1.3, 1.3)
    ax2.yaxis.set_ticks([-1, 0, 1])
    ax2.yaxis.set_ticklabels([0.1, 1, 10])
    ax2.set_ylabel('Absolute value of\nwindow-averaged charge\ndensity $| \\langle \sigma \\rangle _{W} |$, nC cm$^{-2}$')

    # Plotting the Paschen's curve on the same plot
    # 1 kV/mm pultiplied by epsilon_0 (8.65e012 F/m) is 0.885 nC/(cm^2)
    field_to_chargedensity_factor = 0.885 #nC/(cm^2)
    spl3 = pickle.load(open("paschen_air.pickle", "rb"))
    xxs = np.linspace(0.1, 10, 100)
    ax2.plot(xxs, np.log10(
                    field_to_chargedensity_factor*np.power(10, splev(np.log10(xxs), spl3))
                 ),
             'limegreen', lw=3, alpha=0.5,
            label='Empirical interpolating function')

    # from matplotlib import ticker as mticker
    # ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    # ax2.yaxis.set_ticks([np.log10(x) for p in range(-2, 0) for x in np.linspace(10 ** p, 10 ** (p + 1), 10)],
    #                       minor=True)
    # ax2.set_xlim(-1, 1)
    # ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    # ax2.xaxis.set_ticks([np.log10(x) for p in range(-1, 2) for x in np.linspace(10 ** p, 10 ** (p + 1), 10)],
    #                       minor=True)

    # quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    # whiskers = np.array([
    #     adjacent_values(sorted_array, q1, q3)
    #     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    # inds = np.arange(1, len(medians) + 1)
    # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax3 = axarr[2]
    plt.scatter(x=xs, y=ys, color='C3')
    for_export = np.stack([xs, ys])
    np.save(target + '/processed/moving_averaged_net_charge.npy', for_export)

    ax2b = ax2.twinx()
    color = 'limegreen'
    ax2b.set_ylabel('Paschen\'s Law plotted\nas $\epsilon_{0}E_{i}( \sqrt{A} )$, nC cm$^{-2}$', color=color)  # we already handled the x-label with ax1
    ax2b.set_ylim(-1.3, 1.3)
    ax2b.yaxis.set_ticks([-1, 0, 1])
    ax2b.yaxis.set_ticklabels([0.1, 1, 10])
    ax2b.tick_params(axis='y', labelcolor=color)

    if plot_linear_fit:
        cut_from = 8
        cut_to = 20
        xs_for_fit = xs[cut_from:cut_to]
        ys_for_fit = ys[cut_from:cut_to]
        def func_lin(x, a):
            return a * x
        popt_lin, pcov = curve_fit(func_lin, xs_for_fit, ys_for_fit)
        ax3.plot(xs, func_lin(xs, popt_lin[0]), '--', color='C0', label='Linear ($\propto\sqrt{A}$)')

    cut_from = 0
    cut_to = 8
    xs_for_fit = xs[cut_from:cut_to]
    ys_for_fit = ys[cut_from:cut_to]
    def func_sq(x, a):
        return a * x**2
    popt_sq, pcov = curve_fit(func_sq, xs_for_fit, ys_for_fit)
    ax3.plot(xs, func_sq(xs, popt_sq[0]), '--', color='C1', label='Square ($\propto$A)')

    xs_for_fit = np.log(xs)
    ys_for_fit = np.log(ys)
    def func_pl(x, a, b):
        return a + x*b
    popt_pl, pcov = curve_fit(func_pl, xs_for_fit, ys_for_fit)
    ax3.plot(xs, np.exp(func_pl(np.log(xs), popt_pl[0], popt_pl[1])),
             '--', color='black', zorder=-40, label='Power law $\propto(\sqrt{{A}})^{{{0:.2f}}}$'.format(popt_pl[1]))
    print('Power exponent is: {0}'.format(popt_pl[1]))

    ax3.set_yscale('log')
    ax3.set_ylabel('Mean absolute value of\nwindow-integrated charge\n$\\langle A\cdot | \\langle \sigma \\rangle _{W}| \\rangle $, nC')
    ax3.set_xlabel('Window dimension ($\sqrt{A}$), mm')
    ax3.legend()

    ax2.set_xscale('log')
    # # ax2.set_yscale('log')
    # plt.show()
    # f1 = plt.figure(1, figsize=(6, 7))
    # ax = f1.gca()
    # plt.pcolor(X, Y, averaged_mask, cmap='viridis')
    # plt.axis('scaled')
    # plt.xlim([X.min(), X.max()])
    # plt.ylim([Y.min(), Y.max()])
    # plt.colorbar()
    axarr[2].set_xlim(0.099, 10.01)
    plt.tight_layout()
    f2.savefig(target + '/graphs/window_averaging.png', dpi=300)
    plt.show()

def calibrate_SEM(target, saveto='movavg_stats.png', maxval=False):

    data = np.load('{0}/processed/data/data.npy'.format(target))
    position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    sigma_map = np.load(target + '/processed/charge_density_after_bkg_subtracted.npy')
    sigma_mask = np.load(target + '/processed/stamp_mask.npy')

    # f1 = plt.figure(1, figsize=(6, 7))
    # ax = f1.gca()
    # # plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    # #           cmap = 'viridis')
    # if not maxval:
    #     maxval = max(abs(sigma_map.min()), abs(sigma_map.max()))
    # plt.pcolor(X, Y, sigma_map, norm=colors.Normalize(vmin=-1 * maxval, vmax=maxval),
    #            cmap='seismic')
    # plt.axis('scaled')
    # plt.xlim([X.min(), X.max()])
    # plt.ylim([Y.min(), Y.max()])

    from skimage.morphology import rectangle
    from skimage.filters import rank
    from skimage.draw import rectangle as rectangle_draw

    step_size_in_x = (X[0, 1] - X[0, 0])
    step_size_in_y = (Y[1, 0] - Y[0, 0])
    height, width = sigma_map.shape
    #resize to make the pixels square
    new_width = int(round(width*step_size_in_x/step_size_in_y))
    resized_sigma_map = skimage.transform.resize(sigma_map, (height, new_width))
    # #save to file for matching with SEM image
    # im = skimage.exposure.rescale_intensity(resized_sigma_map, out_range='float')
    # skimage.io.imsave(target + '/processed/raw_dump.png', skimage.img_as_uint(im))
    smoothing = 4
    SEM_image = skimage.io.imread(target + '/processed/SEM_overlay.png')[:, :, 0]
    SEM_image_smoothed = skimage.filters.gaussian(SEM_image, sigma=smoothing, preserve_range=True)
    SEM_mask = skimage.io.imread(target + '/processed/SEM_mask.png')[:,:,0]>0
    sigma_list = resized_sigma_map[SEM_mask].flatten()
    SEM_list = SEM_image_smoothed[SEM_mask].flatten()
    # plt.hist2d(sigma_list, SEM_list, bins=100)
    # plt.show()
    xedges = np.linspace(-3, 5, 200)
    yedges = np.linspace(60, 200, 100)
    H, xedges, yedges = np.histogram2d(sigma_list, SEM_list, bins=(xedges, yedges))
    vert_sums = np.max(H, axis=1)
    for i in range(200-1):
        # print(vert_sums[i])
        H[i,:] = H[i,:]/vert_sums[i]
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(131)
    # ax.set_title('imshow: equidistant')

    im = plt.imshow(H.T, interpolation='nearest', origin='low',extent=[xedges.min(),xedges.max(),
                                                                     yedges.min(),yedges.max()],
                    aspect='auto')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.xlabel('Local surface charge density $\sigma$, nC cm$^{-2}$')
    plt.ylabel('SEM image intensity, a.u.')
    # f1.savefig(target + '/processed/SEM_calibration/{0:03d}.png'.format(smoothing))
    # plt.close('all')
    # plt.scatter(sigma_list, SEM_list, s=1, alpha=0.1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Probability normalized by its maximum for a given $\sigma$')
    plt.tight_layout()
    fig.savefig(target + '/processed/SEM_calibration.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    capacitance = epsilon * (8.85e-12) * area / thickness
    # postprocess_folder('experimental_data\\improved_kp_kelvinprobe\\20180801_SEM_B__repeat0_copy',
    #                    bkg_pos=[[100, 30], [-100, -100], []],
    #                    force_zero_shifts = False,
    #                    maxval=4)
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20180801_SEM_B__repeat0_copy',
    #                    bkg_pos=[[80, 50], [130, 360], [970, 50]],
    #                    maxval=2.4)
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20180801_SEM_B__repeat0_copy',
    #                    bkg_pos=[[80, 50], [130, 360], [970, 50]],
    #                    maxval=2.4, do_preview=False)
    # calibrate_SEM('experimental_data\\improved_kp_kelvinprobe\\20180801_SEM_B__repeat0_copy')

    thickness = 0.55e-6  # m
    area = (1e-2) ** 2  # m^2
    capacitance = epsilon * (8.85e-12) * area / thickness
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe' +\
    #     '\\20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
    #                    bkg_pos=[[213, 85], [3383, 107], [3341, 191]],
    #                    maxval=2.4,
    #              do_preview=True, fitting_order=1, do_alignment=True, window_for_max=[0, 3531, 34, 210])
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191017_5cm_3in_62RH_eq30min_oldUVPDMS_PMMAtol_uniformspeeds_0p1_B01 - Copy',
    #                    bkg_pos=[[1814, 191], [243, 43], [3754, 95]],
    #                    maxval=2.4, do_preview=True, window_for_max=[0, 3531, 34, 210])
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191030_5cm_3in_50RH_ambRH38_eq30min_newPDMS5to1_PMMAtol_uniformspeeds_0p5_C01_copy',
    #                    bkg_pos=[[256, 63], [1952, 193], [3235, 46]],
    #                    maxval=2.4, do_preview=True, window_for_max=[0, 3531, 34, 210])
    # get_moving_averaged_stats('experimental_data\\improved_kp_kelvinprobe\\20191030_5cm_3in_50RH_ambRH38_eq30min_newPDMS5to1_PMMAtol_uniformspeeds_0p5_C01_copy',
    #                    was_precomputed=True, violin_kernel_points = 200, plot_linear_fit=False)
    # Power exponent is: 1.719821505739968

    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',
    #                    bkg_pos=[[213, 85], [3383, 107], [3341, 191]],
    #                    maxval=3.4, do_preview=False, window_for_max=[0, 3531, 86, 199]
    #              )
    # get_moving_averaged_stats('experimental_data\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',
    #                    was_precomputed=True, violin_kernel_points = 200)
    # exponent was: 1.9899313486687122

    # plot_bipolar('F:\\PDMS-PMMA_delamination_experiments\\kelvin_probe\\'
    #              '20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',
    #                    bkg_pos=[[499, 30], [3648, 98], [1581, 169], [1781, 179], [499, 179], [3139, 165], [1464, 30], [1764, 15], [2355, 33], [3447, 58],
    #                             [65, 30], [90, 162], [2715, 30]],
    #                    do_preview=True, fitting_order=2,
    #             window_for_max=[0, 3436, 40, 157])
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\'
    #              '20200131__Argonauts_5cm_3in_41RH_2hydrometer_46RH_ambRH25_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p4_B01',
    #                    bkg_pos=[[170, 141], [1750, 65], [1350, 65], [1150, 65], [1000, 65], [600, 65], [300, 65],
    #                             [1962, 190], [2300, 190], [2600, 190], [3000, 190], [3574, 170]],
    #                    maxval=7.0, do_preview=False, fitting_order=1, do_alignment=True)
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\'
    #              '20200123__Argonauts_5cm_3in_64RH_2hydrometer_68RH_ambRH25_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p4_E01',
    #                    bkg_pos=[[3541, 178], [3454, 88], [100, 186]],
    #                    maxval=5.0, do_preview=False, fitting_order=1, do_alignment=False)

    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191107_5cm_3in_41RH_ambRH36_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_B01',
    #                    bkg_pos=[[468, 91], [1666, 237], [3500, 131], [3415, 202], [2408, 90], [75, 160]],
    #                    maxval=2.4, do_preview=False, window_for_max=[118, 3553, 98, 228],
    #              scalebar_loc='upper right')
    # get_moving_averaged_stats('experimental_data\\improved_kp_kelvinprobe\\20191107_5cm_3in_41RH_ambRH36_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_B01',
    #                    was_precomputed=True, violin_kernel_points = 200, plot_linear_fit=False)
    # exponent was: 1.9257662286428656

    # plot_bipolar('Y:\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p6_E01\\kelvinprobe',
    #                    bkg_pos=[[192, 31], [1814, 9], [3562, 22], [3616, 101], [2694, 146], [669, 176]],
    #                    maxval=1.8, do_preview=False, window_for_max=[0, 3669, 4, 172],
    #              scalebar_loc='upper right')
    # get_moving_averaged_stats('Y:\\PDMS-PMMA_delamination_experiments\\data\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p6_E01\\kelvinprobe',
    #                    was_precomputed=True, violin_kernel_points = 200)

    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
    #                    bkg_pos=[[52, 166], [2450, 7], [3288, 23], [1125, 24], [200, 42], [499, 185], [2715, 136], [1782, 160]],
    #                    maxval=1.8, do_preview=False, window_for_max=[130, 3531, 6, 172],
    #              scalebar_loc='upper right')
    # get_moving_averaged_stats('experimental_data\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',
    #                    was_precomputed=True, violin_kernel_points = 200)
    # exponent was: 1.652468406356277

    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191113_70RH_B01',
    #                    bkg_pos=[[277, 46], [1358, 40], [2450, 36], [3245, 65], [3309, 157], [2535, 195], [1461, 200], [213, 198]],
    #                    maxval=1.8, do_preview=True, window_for_max=[0, 3415, 45, 186])
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191118_77RH_B01',
    #                    bkg_pos=[[163, 66], [1435, 48], [2905, 17], [3814, 40], [3739, 145], [2884, 182], [1800, 202], [409, 235]],
    #                    maxval=0.7, do_preview=False, window_for_max=[0, 3934, 27, 226], c_premax=0.93, nbins = 100)
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191108_5cm_3in_31RH_ambRH32_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_E01',
    #                    bkg_pos=[[324, 63], [1525, 61], [2708, 59], [3462, 93], [3526, 198], [2628, 221], [817, 216]],
    #                    maxval=2.3, do_preview=False, window_for_max=[2300, 3563, 76, 210], c_premax=0.86, nbins = 150,
    #                     right_edge_condition=5e-3) #[118, 3563, 76, 210]
    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
    #                    bkg_pos=[[138, 73], [1243, 64], [3494, 89], [3461, 189], [1616, 218], [232, 226]],
    #                    maxval=2.3, do_preview=False, window_for_max=[0, 3669, 74, 214],
    #              scalebar_loc='upper right')
    # get_moving_averaged_stats('experimental_data\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_1',
    #                    was_precomputed=True, violin_kernel_points = 200)
    # Exponent was: 1.5960862386206447

    # plot_bipolar('experimental_data\\improved_kp_kelvinprobe\\'
    #             '20191108_5cm_3in_31RH_ambRH32_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_E01',
    #                    bkg_pos=[[138, 73], [1243, 64], [3494, 89], [3461, 189], [1616, 218], [232, 226]],
    #                    maxval=2.3, do_preview=True, window_for_max=[0, 3669, 74, 214])

    plt.show()
 #   plot_corrected_pics('20190116_Quiang_membrane2', bkg_pos='right',
  #                      cut_from_left=60)
    #plot_bipolar('experiments/sample2_highres_1')
    #plot_bipolar('experiments/sample1_highres_2')
    #plot_bipolar('experiments/sample2_zoom_1', bkg = 0.669282680827785)
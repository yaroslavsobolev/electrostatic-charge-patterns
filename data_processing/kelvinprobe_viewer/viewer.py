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
#
#font = {'size' : 18}
#matplotlib.rc('font', **font)
#matplotlib.rc('font', **{'family':'serif','serif':['Palatino']})

epsilon = 4 # dielectric constant
thickness = 1e-6 #m
area = (1e-2)**2 # m^2
capacitance = epsilon*(8.85e-12)*area/thickness

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

def correct_shifts_for_target(target):
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
        plt.show()
        min_index = np.argmin(mismatches)
        if min_index == 0 or min_index == len(mismatches)-1 or ((max(mismatches) - min(mismatches)) < 0.001):
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
    f1.savefig(main_folder + '\\graphs\\WF_and_GD_processed.png', dpi = 300)
    f1.clear()
    f1.clf()     

##correct_shifts_for_target('experiments/sample2_zoom_1')



def plot_corrected_pics(target, bkg = 0):
    data = np.load('{0}/processed/data/data.npy'.format(target))
    position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    pot = -1*data[1]/1000
    if not bkg:
        bkg = np.mean(pot[:10, :])
        print('Background is {0}'.format(bkg))
    pot = pot - bkg
    
    f1 = plt.figure(1)
    ax = f1.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    maxval = max(abs(pot.min()), abs(pot.max()))
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval),
               cmap = 'seismic')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    plt.colorbar()
    f1.savefig(target + '/processed/final_fig_bipolar.png', dpi=600)
    plt.axis('off')
    f1.savefig(target + '/processed/final_fig_bipolar_noaxis.png', dpi=600)
    f1.clf()
    
    f3 = plt.figure(3, figsize = (6,6))
    ax = f3.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=0, vmax=pot.max()),
               cmap = 'viridis')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    plt.colorbar()
    f3.savefig(target + '/processed/final_fig_negative_part.png', dpi=600)
    plt.axis('off')
    f3.savefig(target + '/processed/final_fig_negative_part_noaxis.png', dpi=600)
    f3.clf()
                                                                                                                                                                                                                                                                                                                                                    
#plot_corrected_pics('experiments/sample2_zoom_1', bkg = 0.669282680827785)

def plot_bipolar(target, saveto = 'bipolar.png', processed = True, bkg=0):
    if processed:
        data = np.load('{0}/processed/data/data.npy'.format(target))
        position_griddata = np.load('{0}/processed/data/position_griddata.npy'.format(target))
    else:
        data = np.load('{0}/data/data.npy'.format(target))
        position_griddata = np.load('{0}/data/position_griddata.npy'.format(target))
    X,Y = position_griddata
    pot = -1*data[1]/1000
    if not bkg:
        bkg = np.mean(pot[:10, :])
        print('Background is {0}'.format(bkg))
    pot = pot - bkg
    f1 = plt.figure(1, figsize=(6,7))
    ax = f1.gca()
    #plt.pcolor(X,Y,data[1], norm=colors.Normalize(vmin=data[1].min(), vmax=data[1].max()/8),
    #           cmap = 'viridis')
    maxval = max(abs(pot.min()), abs(pot.max()))
    plt.pcolor(X,Y,pot, norm=colors.Normalize(vmin=-1*maxval, vmax=maxval),
               cmap = 'seismic')
    plt.axis('scaled')
    plt.xlim([X.min(), X.max()])
    plt.ylim([Y.min(), Y.max()])
    
    scalebar = ScaleBar(1 ,'mm', location = 'lower right', font_properties={'size':0.03}, sep=-1)
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
    f1.savefig(target + '/' + saveto, dpi=600)
    f1.clf()

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

#make_xyz_files('experiments/sample1_highres_2')
#make_xyz_files('experiments/sample2_highres_1')
#make_xyz_files('experiments/sample2_zoom_1')

#plot_bipolar('experiments/sample1_lowres', processed = False)
#plot_bipolar('experiments/sample2_highres_1')
#plot_bipolar('experiments/sample1_highres_2')
plot_bipolar('experiments/sample2_zoom_1', bkg = 0.669282680827785)
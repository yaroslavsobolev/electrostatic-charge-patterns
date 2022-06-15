# -*- coding: utf-8 -*-
"""
Software that controls the improvements made on top
of the KP scanning Kelvin probe system.
The improvements mainly compensate for the poor
choice of linear stages made by KP engineers.
The result is 5x improvement in scanning speed
(and more if the scan steps are larger).

Created on Tue May  8 10:08:02 2018

@author: Yaroslav I. Sobolev
"""

import kpapi
import time
import numpy as np
import uuid
import thorlabs_apt as apt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

center_of_circle_holder = (18.5, 59)
flatness_angles = (7.10631, 7.18563)
flatness_angles_2= (7.19642, 7.00533)
flatness_angles_3= (7.13092, 7.21779)
flatness_angles_4= (6.97968, 7.01743)

a = apt.list_available_devices()
print(a)
#angle_x = apt.Motor(83859943)
y_stage = apt.Motor(67862050)
x_stage = apt.Motor(83858432)
angle_stage_x = apt.Motor(83859943)
angle_stage_y = apt.Motor(83855305)
#angle_x.move_home()
#y_stage.move_home()
#time.sleep(10)
#while not y_stage.has_homing_been_completed:
#    print('Homing...')
#    time.sleep(0.5)
#print('..Homing done.')
kapi = kpapi.KP_software_API()
def move_stage_and_wait(stage, pos, delay=0.02, verbose=False):
    if stage == 'y':
        y_stage.move_to(pos)
        time.sleep(delay)
        while (not y_stage.is_settled) or y_stage.is_in_motion:
            if verbose:
                print('Stage Y: is_tracking:{0}, is_settled:{1}, is_in_motion:{2}'.format(
                        y_stage.is_tracking,
                        y_stage.is_settled,
                        y_stage.is_in_motion))
            time.sleep(0.01)
#        print('Stage Y: ...moving done.')
    elif stage == 'x':
        x_stage.move_to(pos)
        time.sleep(delay)
        while x_stage.is_in_motion:
            if verbose:
                print('Stage X: is_tracking:{0}, is_settled:{1}, is_in_motion:{2}'.format(
                        x_stage.is_tracking,
                        x_stage.is_settled,
                        x_stage.is_in_motion))
            time.sleep(0.01)
#        print('Stage X: ...moving done.')
    else:
        raise Exception
    
def move_to_xy_and_wait(xypos):
    x,y = xypos
    move_stage_and_wait('y', y)
    move_stage_and_wait('x', x)

def move_in_sequence(begin, end, N_pts, wait_at_each_point_for = 2):
    position_sequence = []
    for x in np.linspace(begin[0], end[0], N_pts):
        for y in np.linspace(begin[1], end[1], N_pts):
            position_sequence.append([x,y])
    position_griddata = np.meshgrid(np.linspace(begin[0], end[0], N_pts),
                        np.linspace(begin[1], end[1], N_pts))
    position_sequence = np.array(position_sequence)
    time_sequence = []
    t0 = time.time()
    for p in position_sequence:
        print('MOVING TO: {0}'.format(p))
        move_to_xy_and_wait(p)
        time_sequence.append(time.time()-t0)
        time.sleep(wait_at_each_point_for)
    return position_sequence, time_sequence, position_griddata

def move_in_line(begin, end, N_pts, axis, wait_at_each_point_for = 2):
    position_sequence = []
    if axis == 'x':
        y = begin[1]
        for x in np.linspace(begin[0], end[0], N_pts):
    #        for y in np.linspace(begin[1], end[1], N_pts):
            position_sequence.append([x,y])
    elif axis == 'y':
        x = begin[0]
        for y in np.linspace(begin[1], end[1], N_pts):
            position_sequence.append([x,y])
    position_sequence = np.array(position_sequence)
    time_sequence = []
    t0 = time.time()
    for p in position_sequence:
        print('MOVING TO: {0}'.format(p))
        move_to_xy_and_wait(p)
        time_sequence.append(time.time()-t0)
        time.sleep(wait_at_each_point_for)
    return position_sequence, time_sequence

def extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence, dt = 1):
    time_sequence = np.array(time_sequence)
    data = np.genfromtxt(filename, skip_header=1, skip_footer=21, delimiter=',')
    nrows, ncols = data.shape
    new_data = np.zeros(shape=(len(time_sequence), ncols), dtype = np.float64)
    for colid in range(ncols):
        new_col = np.interp(time_sequence+dt, data[:,-1], data[:,colid])
        new_data[:,colid] = new_col
    return new_data

def measure_wf_and_gd_at_this_point():
#    kapi = 
    kapi.start_wf_measurement()
    time.sleep(3)
    uuid_here = uuid.uuid4().hex
    filename = 'temp\\{0}.dat'.format(uuid_here)
    kapi.stop_wf_measurement()
    time.sleep(0.5)
    kapi.save_measurement_file(filename)
    time_sequence = np.linspace(1,2,10)
    data = extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence)
    wf = np.mean(data[:,1])
    gd = np.mean(data[:,5])
    return (wf, gd)

def approach_to_target_grad(target_gd):
    print('Moving to gradient value {0:.2f}'.format(target_gd))
#    kapi = kpapi.KP_software_API()
    gd = measure_wf_and_gd_at_this_point()[1]
    i = 0
    while gd < target_gd:
        i += 1
        if i > 10:
            break
        if gd < 30:
            step = 400
        elif gd < 40:
            step = 200
        elif gd < 50:
            step = 50
        elif gd < 70:
            step = 10
        else:
            step = 1
        print('Current gd: {0:.2f}, stepping by {1:.2f}'.format(gd, step))
        kapi.move_z_stage(step)
        time.sleep(3)
        gd = measure_wf_and_gd_at_this_point()[1]

def measure_at_linepoints(begin, end, N_pts, axis):
#    kapi = kpapi.KP_software_API()
    kapi.start_wf_measurement()
    position_sequence, time_sequence = move_in_line(begin, end, N_pts, axis)
    uuid_here = uuid.uuid4().hex
    filename = 'temp\\{0}.dat'.format(uuid_here)
    kapi.stop_wf_measurement()
    time.sleep(0.5)
    kapi.save_measurement_file(filename)
    data = extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence)
    return position_sequence, data, time_sequence

def measure_at_gridpoints(begin, end, N_pts):
#    kapi = kpapi.KP_software_API()
    kapi.start_wf_measurement()
    position_sequence, time_sequence, position_griddata = move_in_sequence(begin, end, N_pts)
    uuid_here = uuid.uuid4().hex
    filename = 'temp\\{0}.dat'.format(uuid_here)
    kapi.stop_wf_measurement()
    time.sleep(0.5)
    kapi.save_measurement_file(filename)
    data = extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence)
    return position_sequence, data, time_sequence, position_griddata

def one_iteration_of_angular_alignment(begin, end, N_pts):
    position_sequence, data, time_sequence, position_griddata = measure_at_gridpoints(begin, end, N_pts)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xs=position_sequence[:,0], ys=position_sequence[:,1], zs=data[:,5], c = data[:,5])
    z_min = np.min(data[:,5])
    for i, z in enumerate(data[:,5]):
        xs = [position_sequence[i,0], position_sequence[i,0]]
        ys = [position_sequence[i,1], position_sequence[i,1]]
        zs = [z_min, z]
        ax.plot(xs, ys, zs, c = 'black')
    X1, X2 = position_griddata
    ncols = X1.shape[0]
    Y = np.reshape(data[:,5], (ncols, ncols)).T
    #ax.plot_surface(X1, X2, Y)
    m = ncols
    X = np.hstack( (np.reshape(X1, (m*m, 1)) , np.reshape(X2, (m*m, 1)) ) )
    X = np.hstack( ( np.ones((m*m, 1)), X ))
    YY = np.reshape(Y, (m*m, 1))
    
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    print(theta)
    plane = np.reshape(np.dot(X, theta), (m, m))
    
    ax.plot_surface(X1, X2, plane)
    plt.show()
    
    return position_sequence, data, time_sequence, position_griddata, theta

    # coefficient for angle stage X is -0.19411046337312196
    # coefficient for angle stage Y is 0.5249738384537345
    # new coefficient for Y plane: 0.043794085842041926
    
    #y_angle = angle_stage_y.position
#    angle_stage_x.move_by((-1)*theta[1]*(-0.004411046337312196))
    #angle_stage_y.move_by((-1)*theta[2]*(0.0043794085842041926))

#position_sequence, data, time_sequence, position_griddata, theta = \
#    one_iteration_of_angular_alignment([9, 52.5], [20.5, 65.5], 2)
    
def quickly_move_in_sequence(begin, end, N_pts_x, N_pts_y):
    t0 = time.time()
    position_sequence = []
    for x in np.linspace(begin[0], end[0], N_pts_x):
        for y in np.linspace(begin[1], end[1], N_pts_y):
            position_sequence.append([x,y])
    position_griddata = np.meshgrid(np.linspace(begin[0], end[0], N_pts_x),
                        np.linspace(begin[1], end[1], N_pts_y))
    position_sequence = np.array(position_sequence)
    time_sequence = []
    print('Moving to the beginning of the scan.')
    move_stage_and_wait('y', begin[1], delay=0.3)
    move_stage_and_wait('x', begin[0])
    time.sleep(3)
    for i,x in enumerate(np.linspace(begin[0], end[0], N_pts_x)):
        print('Moving x stage to another line')
        move_stage_and_wait('x', x)
        time.sleep(1)
        print('Starting Y movement (line {0})'.format(i))
        t1 = time.time()-t0
        if i % 2 == 0:
#        move_stage_and_wait('y', begin[1], delay=1)
            move_stage_and_wait('y', end[1], delay=1)
        else:
            move_stage_and_wait('y', begin[1], delay=1)
        t2 = time.time()-t0
        print('Ended Y movement')
        if i % 2 == 0:
            time_sequence.extend(list(np.linspace(t1, t2, N_pts_y)))
        else:
            time_sequence.extend(list(reversed(np.linspace(t1, t2, N_pts_y))))
        
#            time.sleep(wait_at_each_point_for)
    return position_sequence, time_sequence, position_griddata

#position_sequence, time_sequence, position_griddata = quickly_move_in_sequence([12, 46], [13, 56], 3, 100)

# scan a square area under 10000 points

def scan_small_fragment(begin, end, N_pts_x, N_pts_y, destination, verbose = True):
    if not os.path.exists(destination):
        os.makedirs(destination)
    freq_kp = 5
    kp_points = (end[1] - begin[1])/y_stage.maximum_velocity*freq_kp
    if verbose:
        print('Each line will take {0:.2f} seconds to scan.'.format((end[1] - begin[1])/y_stage.maximum_velocity))
        print('Kelvin probe will acquire approximately {0:.0f} points in each line, '
              'which will then be resampled into {1:.0f} points.'.format(kp_points, N_pts_y))
        print('This measurement will take approximately {0:.1f} minutes.'.format(
                (end[1] - begin[1])/y_stage.maximum_velocity*N_pts_x/60))
    kapi = kpapi.KP_software_API()
    kapi.start_wf_measurement()
    position_sequence, time_sequence, position_griddata = \
                        quickly_move_in_sequence(begin, end, N_pts_x, N_pts_y)
    uuid_here = uuid.uuid4().hex
    filename = '{0}\\{1}.dat'.format(destination, uuid_here)
    kapi.stop_wf_measurement()
    time.sleep(0.5)
    kapi.save_measurement_file(filename)
    data = extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence, dt=0)
    
    f1 = plt.figure(1, figsize=(4,8))
    ax = plt.subplot(211)
    plt.title('WF')
    Y = np.reshape(data[:,1], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
    plt.imshow(Y)
    plt.colorbar()
    ax = plt.subplot(212)
    plt.title('Gradient')
    Y = np.reshape(data[:,5], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
    plt.imshow(Y)
    plt.colorbar()
#    f1.savefig(destination + '/plots.png', dpi = 1000)
    f1.clear()
    f1.clf()
    
    np.savetxt(destination + '/position_sequence.txt', position_sequence)
    np.savetxt(destination + '/time_sequence.txt', np.array(time_sequence))
    np.savetxt(destination + '/data.txt', data)
    #np.savetxt(destination + '/position_griddata.txt', position_griddata)
    np.save(destination + '/position_sequence.npy', position_sequence)
    np.save(destination + '/time_sequence.npy', np.array(time_sequence))
    np.save(destination + '/data.npy', data)
    np.save(destination + '/position_griddata.npy', position_griddata)

    return position_sequence, data, time_sequence, position_griddata


#y_stage.maximum_velocity = 100
#y_stage.move_to(52)
#time.sleep(1)
#delay = 0.3
#y_stage.maximum_velocity = 0.15
#y_stage.move_to(62)
#time.sleep(delay)
#positions = []
#verbose = True
#while (not y_stage.is_settled) or y_stage.is_in_motion:
#    if verbose:
#        print('Stage Y: is_tracking:{0}, is_settled:{1}, is_in_motion:{2}'.format(
#                y_stage.is_tracking,
#                y_stage.is_settled,
#                y_stage.is_in_motion))
#    positions.append(y_stage.position)
#    time.sleep(0.01)
#plt.plot(positions)
#plt.show()

# scan_large_area()
def scan_large_area(begin, end, N_pts_x, N_pts_y, x_piece_size, maxvel, main_folder):
    #x_piece_size = 20
    y_stage.maximum_velocity = maxvel
#    main_folder = 'doubleQR_highres_1'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(main_folder + '\\fragments'):
        os.makedirs(main_folder + '\\fragments')
    if not os.path.exists(main_folder + '\\graphs'):
        os.makedirs(main_folder + '\\graphs')
    if not os.path.exists(main_folder + '\\data'):
        os.makedirs(main_folder + '\\data')
    x_sequence = np.linspace(begin[0], end[0], N_pts_x)
    pieces_of_x_sequence = np.array_split(x_sequence, range(x_piece_size, N_pts_x, x_piece_size))
    for i,p in enumerate(pieces_of_x_sequence):
        print('Starting to scan fragment {0}'.format(i))
        x_begin = p[0]
        x_end   = p[-1]
        folder_for_fragment = '{0}\\fragments\\{1:05d}'.format(main_folder, i)
        position_sequence, data, time_sequence, position_griddata = \
                scan_small_fragment([x_begin, begin[1]], 
                                    [x_end, end[1]], 
                                    x_piece_size, 
                                    N_pts_y, 
                                    folder_for_fragment)
        data3d = []
        for k in range(data.shape[1]):    
            Y = np.reshape(data[:,k], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
            data3d.append(Y.T)
        data3d = np.array(data3d)
        if i == 0:
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
        plt.close("all")
    
#        f1 = plt.figure(1, figsize=(4,8))
#        ax = plt.subplot(211)
#        plt.title('WF')
#        #    Y = np.reshape(data[:,1], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
#        plt.pcolor(total_position_grid[0], total_position_grid[1], total_data[1])
#        #plt.axes().set_aspect('equal')
#        plt.axis('scaled')
#        plt.colorbar()
#        ax = plt.subplot(212)
#        plt.title('Gradient')
#        plt.pcolor(total_position_grid[0], total_position_grid[1], total_data[5])
#        #    Y = np.reshape(data[:,5], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
#        plt.axis('scaled')
#        #plt.axes().set_aspect('equal')
#        plt.colorbar()
##        f1.savefig(main_folder + '\\graphs\\WF_and_GD_raw.png', dpi = 1000)
#        f1.clear()
#        f1.clf()                                                                                                            
            
#start_1 = 10.00 + (23.0-10)*3/4   
#scan_large_area([start_1, 55], [23, 68], 100, 1000, 20, 0.15, 'SEMed1_continue')


#scan_large_area([0, 40], [25, 80], 200, 3000, 10, 0.2, 
#                'XXXXX')

# [12.3, 35], [22.7, 80],
if __name__ == "__main__":
    scan_large_area([2, 15], [35, 88], 200, 4000, 10, 0.2, 
                    '20220323_5cm_38RHdig_speeds_0p4_B01_run1')

#scan_large_area([10, 51], [21, 64], 400, 3000, 10, 0.15, 
#                '20190930_5cm_62RH_eq15min_humidPDMS_PMMAtol_dz01_B01_run3zoom')

#scan_large_area([10, 55], [23, 68], 400, 1000, 20, 0.15, 'SEMed1_repeat1')
##approach_to_target_grad(60)
#begin = [center_of_circle_holder[0]-15, center_of_circle_holder[1]]
#end = [center_of_circle_holder[0]+15, center_of_circle_holder[1]]
#position_sequence, data, time_sequence = \
#    measure_at_linepoints(begin, end, 50, axis='x')
    

#data3d = []
#for k in range(data.shape[1]):    
#    Y = np.reshape(data[:,k], (position_griddata[0].shape[1], position_griddata[0].shape[0]))
#    data3d.append(Y.T)
#data3d = np.array(data3d)

#ddd = np.concatenate((data3d, data3d), axis=2)

#shifted_posgrid = np.copy(position_griddata)
#shiftt = np.max(position_griddata, axis=(1,2))[0] - np.min(position_griddata, axis=(1,2))[0]
#for i in range(shifted_posgrid.shape[1]):
#    for j in range(shifted_posgrid.shape[2]):        
#        shifted_posgrid[0,i,j] += shiftt
#xxx = np.concatenate((position_griddata, shifted_posgrid), axis=2)
#yyy = np.concatenate((Y.T, Y.T), axis=1)
#X1, X2 = xxx
#plt.pcolor(X1, X2, yyy)
#plt.axes().set_aspect('equal')
#plt.show()

#x_stage.maximum_velocity

#def quickly_measure_at_gridpoints(begin, end, N_ptsx, N_ptsy):
#    kapi = kpapi.KP_software_API()
#    kapi.start_wf_measurement()
#    position_sequence, time_sequence, position_griddata = move_in_sequence(begin, end, N_pts)
#    uuid_here = uuid.uuid4().hex
#    filename = 'temp\\{0}.dat'.format(uuid_here)
#    kapi.stop_wf_measurement()
#    time.sleep(1)
#    kapi.save_measurement_file(filename)
#    data = extract_from_datfile_and_interpolate_at_timepoints(filename, time_sequence)
#    return position_sequence, data, time_sequence, position_griddata
# -*- coding: utf-8 -*-
"""

Script for continuous saving the data from the electronic 
balance (model: XPE205 Mettler Toledo)

Created on Fri Sep 22 22:52:55 2017

Author: Yaroslav I. Sobolev
"""

#import configparser
import visa
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import thorlabs_apt as apt
import numpy as np
import os
import re
import json
from scipy import signal
plt.ion()

#def start_electrometer_recording(visaresourcemanager, nreads = 50000):   
#    keithley = visaresourcemanager.open_resource('GPIB0::27::INSTR')
#    print(keithley.write(":SYST:ZCH 0"))
#    print(keithley.write(":TRACE:CLE"))
#    print(keithley.write(":TRAC:POINTS {0}".format(nreads)))
#    print(keithley.write(":SENSE:FUNC 'CHAR'"))
#    print(keithley.write(":SENSE:CHAR:RANGE 2e-7"))
#    print(keithley.write(":SENSE:CHAR:AVER:STAT OFF"))
#    print(keithley.write(":SENSE:CHAR:MED:STAT OFF"))
#    print(keithley.write(":SENSE:CHAR:DIGits 7"))
#    print(keithley.write(":TRIG:COUNT {0}".format(nreads)))
#    print(keithley.write(":TRIG:DEL 0.0"))
#    print(keithley.write(":SENSE:CHAR:NPLC 0.01"))
#    print(keithley.write(":FORM:ELEM RNUM, READ, TST"))
#    print(keithley.write(":CALC:STAT OFF"))
#    print(keithley.write(":SYSTEM:LSYN:STAT OFF"))
#    print(keithley.write(":TRAC:FEED:CONT NEXT"))
#    print(keithley.write(":SYST:ZCH OFF"))
#    print(keithley.write(":INIT"))
#    return keithley
#
#def download_data_from_keithley(keithley):
#    print("Downloading the data from the Keithley electrometer...")
#    regex1 = re.compile(r'''\+(?P<reading>\d+?),(?P<value>[+-](?:[.\d])+?)E-09,(?P<datetime>\d\d:\d\d:\d\d.00,\d\d-\S\S\S-\d\d\d\d)''')
#    keithley_response = keithley.query(":TRACE:DATA?")
#    data = regex1.findall(keithley_response)
#    first_read_time = datetime.strptime(data[0][2], '%H:%M:%S.00,%d-%b-%Y')
#    last_read_time = datetime.strptime(data[-1][2], '%H:%M:%S.00,%d-%b-%Y')
#    delta = last_read_time - first_read_time
#    delta_seconds = delta.seconds - 1
#    first_sec_change = next(i for i in data if i[2] != data[0][2])
#    last_sec_change = next(i for i in reversed(data) if i[2] != data[-1][2])
#    persec = (int(last_sec_change[0]) - int(first_sec_change[0]))/delta_seconds
#    print('..download was successful. Speed was {0} readings per second.'.format(persec))
#    return data, persec

## ============== connection to motor 
print(apt.list_available_devices())
#z_stage = apt.Motor(27502436)
z_stage = apt.Motor(83855315)
z_stage.backlash_distance = 0
z_stage.acceleration = 4.5
position = z_stage.position
init_position = position
dt = 0.1
base_path = 'saved_data/'
experiment_name = input('Enter filename for saving the data:')
dirname = 'data/' + experiment_name
# create directories for saving the data
try:
    os.mkdir(dirname)
    print("Directory " , dirname ,  " created ") 
except FileExistsError:
    print("Directory " , dirname ,  " already exists")
    
try:
    os.mkdir(dirname + '/force')
    os.mkdir(dirname + '/electrometer')
    os.mkdir(dirname + '/video')
    os.mkdir(dirname + '/kelvinprobe')
    os.mkdir(dirname + '/stamp')
    print("Directories under " , dirname ,  " created.") 
except FileExistsError:
    print("Directorise under " , dirname ,  " already exist.")

f = open(dirname + '/force/force_vs_time.txt', 'w+')
f.write('Time_elapsed(sec)\tZ-position(mm)\tWeight(g)\tTimestamp\r\n')
f.close()
## ============= Connection to balance
rm = visa.ResourceManager()
print(rm.list_resources())
balance = rm.open_resource('ASRL8::INSTR', baud_rate = 38400, data_bits = 8, stop_bits = visa.constants.StopBits.one)
start_time = datetime.now()
fig = plt.figure(1)
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time elapsed (seconds)', fontsize=12)
plt.ylabel('Weight (grams)', fontsize=12)
weight = 0
step = 0
constant_force_mode = False
switched_to_delamination_speed = False
maximum_weight_so_far = -1000

# ==== PARAMETERS THAT ARE USER-USEFUL ====
initial_speed = 0.005
speed_after_jump = 0.4
arming_force_threshold = 40 # THE VALUE USED IN 2020
#arming_force_threshold = 2.5
threshold_to_const_force = 2
electrometer_starting_force_threshold = 40
# END OF PARAMETERS THAT ARE USER-USEFUL

# saving parameters to JSON file for future reference
json_dict = {'initial_speed': initial_speed,
             'speed_after_jump': speed_after_jump,
             'arming_force_threshold':arming_force_threshold,
             'threshold_to_const_force':threshold_to_const_force,
             'electrometer_starting_force_threshold':electrometer_starting_force_threshold,
             }
with open(dirname + '/force/parameters.txt', 'w') as json_file:
    json.dump(json_dict, json_file)
constant_force_step = 0
electrometer_was_started = False
electrometer_start_time = datetime.now()
z_stage.maximum_velocity = initial_speed
time.sleep(0.2)
z_stage.move_to(5)
time.sleep(0.2)
while(step < 1500):
    step += 1
    if (maximum_weight_so_far - weight > threshold_to_const_force) and \
                        (maximum_weight_so_far > arming_force_threshold):
        constant_force_mode = True
    if (not electrometer_was_started) and \
            (weight > electrometer_starting_force_threshold):
#        keithley = start_electrometer_recording(rm, nreads = 50000)
        electrometer_start_time = datetime.now()
        electrometer_was_started = True 
    if constant_force_mode:
        print('Constant force mode ON.')
        constant_force_step += 1
        if not switched_to_delamination_speed:
            z_stage.stop_profiled()
            time.sleep(0.1)
            z_stage.maximum_velocity = speed_after_jump
            time.sleep(0.1)
            z_stage.move_to(5)
            switched_to_delamination_speed = True
        
        if (not electrometer_was_started):
#            keithley = start_electrometer_recording(rm, nreads = 50000)
            electrometer_start_time = datetime.now()
            electrometer_was_started = True 
    position = z_stage.position
    print('Z position: {0}'.format(position))
    time.sleep(dt)
    try:
        response = balance.query("SI")
        weight = float(response[5:-4])
    except:
        print('Responce from the balance does not contain weight in grams. Check if the balance is working properly.')
        time.sleep(1)
        continue
    time_elapsed = datetime.now() - start_time
    plt.title('Operation frequency: {0:.1f} fps'.format(step/time_elapsed.total_seconds()))
    with open(dirname + '/force/force_vs_time.txt', 'a') as f:
        f.write('{0:.2f}\t{1}\t{2}\t{3}\n'.format(time_elapsed.total_seconds(),
                                          position,
                                          response[5:-4], datetime.now().strftime("%H:%M:%S / %d %B %Y")))
    plt.plot(time_elapsed.total_seconds(), weight, color = 'blue', marker = "o", ls="")
    plt.draw()
    
    if weight > maximum_weight_so_far:
        maximum_weight_so_far = weight
    plt.pause(0.05)

    # stop recording force if the weight is negative and 500 steps were done
    if step > 500 and weight < 0:
        break

## wait until electrometer finishes recording
#electrometer_recording_length = 50000/439 # in seconds
#while True:
#    time.sleep(1)
#    time_elapsed = datetime.now() - electrometer_start_time
#    seconds_elapsed = time_elapsed.total_seconds()
#    if seconds_elapsed > electrometer_recording_length + 10:
#        break
## download_data_from_electrometer
#data, persec = download_data_from_keithley(keithley)
#values = [float(d[1]) for d in data]
#fname =  dirname + '/electrometer/' + "{:%d-%m-%Y_%H-%M}__".format(datetime.now()) + '{0:.2f}rps__'.format(persec)
#np.savetxt(fname + 'all.txt', data, delimiter="\t", fmt="%s")
#np.savetxt(fname + 'values_only.txt', values)
#np.savetxt(dirname + '/electrometer/readings_per_second.txt', np.array([persec]))
#plt.tight_layout()

#fig_electrometer = plt.figure(3)
#time_points = np.linspace(0, len(values)/persec, len(values))
#plt.plot(time_points, values, alpha = 0.9)
#smoothed = signal.savgol_filter(values, 51, 2)
#plt.plot(time_points, smoothed, alpha = 0.7, color = 'r')
#plt.tight_layout()
#plt.xlabel('Time (seconds)')
#plt.ylabel('Net mirror charge on substrate (nC)')
#smoothed = signal.savgol_filter(values, 301, 2)
#plt.plot(smoothed, alpha = 0.5, color = 'r')
#fig_electrometer.savefig(dirname + '/electrometer/electrometer_graph.png', dpi=600)
fig.savefig(dirname + '/force/force_vs_time.png', dpi=600)
time.sleep(1)
z_stage.stop_profiled()
time.sleep(1)
z_stage.maximum_velocity = 2.3
time.sleep(1)
z_stage.move_to(41)
time.sleep(1)
#keithley.close()
balance.close()
electrometer_to_balance_start_time = electrometer_start_time-start_time
np.savetxt(dirname + '/electrometer/electrometer_to_balance_delay.txt', 
           np.array([electrometer_to_balance_start_time.total_seconds()]))
plt.show()
print('--= EXPERIMENT HAS ENDED SUCCESSFULLY =--')
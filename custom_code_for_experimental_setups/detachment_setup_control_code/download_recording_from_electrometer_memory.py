# -*- coding: utf-8 -*-
"""

This script will download the data from Keithey 6517B electrometer memory, 
if the attempt for downloading during normal experimental sequence
failed for some reason.

Created on Tue Nov 22 19:40:55 2019

Author: Yaroslav I. Sobolev
"""

#import configparser
import visa
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
from scipy import signal
plt.ion()

rm = visa.ResourceManager()
print('This script will download the data from electrometer memory, if the last attempt for downloading failed for some reason.')
experiment_name = input('Enter experiment name for saving the data:')
dirname = 'data/' + experiment_name
keithley = rm.open_resource('GPIB0::27::INSTR')

def download_data_from_keithley(keithley):
    print("Downloading the data from the Keithley electrometer...")
    regex1 = re.compile(r'''\+(?P<reading>\d+?),(?P<value>[+-](?:[.\d])+?)E-09,(?P<datetime>\d\d:\d\d:\d\d.00,\d\d-\S\S\S-\d\d\d\d)''')
    keithley_response = keithley.query(":TRACE:DATA?")
    data = regex1.findall(keithley_response)
    first_read_time = datetime.strptime(data[0][2], '%H:%M:%S.00,%d-%b-%Y')
    last_read_time = datetime.strptime(data[-1][2], '%H:%M:%S.00,%d-%b-%Y')
    delta = last_read_time - first_read_time
    delta_seconds = delta.seconds - 1
    first_sec_change = next(i for i in data if i[2] != data[0][2])
    last_sec_change = next(i for i in reversed(data) if i[2] != data[-1][2])
    persec = (int(last_sec_change[0]) - int(first_sec_change[0]))/delta_seconds
    print('..download was successful. Speed was {0} readings per second.'.format(persec))
    return data, persec

# download_data_from_electrometer
data, persec = download_data_from_keithley(keithley)
values = [float(d[1]) for d in data]
fname =  dirname + '/electrometer/' + "{:%d-%m-%Y_%H-%M}__".format(datetime.now()) + '{0:.2f}rps__'.format(persec)
np.savetxt(fname + 'all.txt', data, delimiter="\t", fmt="%s")
np.savetxt(fname + 'values_only.txt', values)
np.savetxt(dirname + '/electrometer/readings_per_second.txt', np.array([persec]))

fig_electrometer = plt.figure(3)
time_points = np.linspace(0, len(values)/persec, len(values))
plt.plot(time_points, values, alpha = 0.9)
smoothed = signal.savgol_filter(values, 51, 2)
plt.plot(time_points, smoothed, alpha = 0.7, color = 'r')
plt.tight_layout()
plt.xlabel('Time (seconds)')
plt.ylabel('Net mirror charge on substrate (nC)')
#smoothed = signal.savgol_filter(values, 301, 2)
#plt.plot(smoothed, alpha = 0.5, color = 'r')
fig_electrometer.savefig(dirname + '/electrometer/electrometer_graph.png', dpi=600)
time.sleep(1)
keithley.close()
plt.show()
print('--= EXPERIMENT HAS ENDED SUCCESSFULLY =--')
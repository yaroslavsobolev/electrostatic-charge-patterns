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
import re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import thorlabs_apt as apt
#from scipy import signal
plt.ion()

base_path = 'saved_data/'
#
#filename = input('Enter filename for saving the data:')
#interval = int(input("Enter interval between measurements (in seconds):"))

#filename = 'test'
interval = 1

#f = open(base_path + filename + '.txt', 'w+')
#f.write('Time_elapsed(sec)\tZ-position(mm)\tWeight(g)\tTimestamp\r\n')
#f.close()

### ============== connection to motor 
#print(apt.list_available_devices())
#z_stage = apt.Motor(27502436)
##z_stage.move_home(False)
##time.sleep(20)
#position = z_stage.position
#init_position = position
#dz = 0.01
dt = 0.3

## ============= Connection to balance
rm = visa.ResourceManager()
print(rm.list_resources())


#
balance = rm.open_resource('ASRL8::INSTR', baud_rate = 38400, data_bits = 8, stop_bits = visa.constants.StopBits.one)
#response = balance.query("SI")
#balance.write('SI')
#print(balance.read())
#print(response)
#balance.close()

##print('1')
##print(balance.query("*IDN?"))
##
start_time = datetime.now()

#f, ax = plt.subplots(1, figsize=(8, 6))
#ax.set_ylim([ymin,ymax])
#plt.axis([0, 10, 0, 1])
#line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time elapsed (seconds)', fontsize=12)
plt.ylabel('Weight (grams)', fontsize=12)
weight = 0
while(weight < 100):
#    position += dz
#    z_stage.move_to(position)
    time.sleep(dt)
    try:
        response = balance.query("SI")
        weight = float(response[5:-4])
    except:
        print('Responce from the balance does not contain weight in grams. Check if the balance is working properly.')
#        print(response)
        time.sleep(interval)
        continue
    print(response)
    print('Weight={0}'.format(response[5:-4]))
    time_elapsed = datetime.now() - start_time
#    with open(base_path + filename + '.txt', 'a') as f:
#        f.write('{0:.2f}\t{1}\t{2}\t{3}\n'.format(time_elapsed.total_seconds(),
#                                          position,
#                                          response[5:-4], datetime.now().strftime("%H:%M:%S / %d %B %Y")))
    plt.plot(time_elapsed.total_seconds(), weight, color = 'blue', marker = "o", ls="")
    plt.title('Weight={0}'.format(response[5:-4]),
              fontdict={'fontsize': 25})
    plt.draw()
    plt.pause(0.05)
#    time.sleep(interval)
#z_stage.move_to(init_position)
balance.close()

# -*- coding: utf-8 -*-
"""
Since KP does not supply API for their software
for scanning Kelvin probe, I've written an 
API by simulating the mouse clicks and keyboard input
at the KP software GUI. I can't believe I'm doing this shit.

Created on Tue May  7 10:08:02 2018
@author: Yaroslav I. Sobolev
"""
import os
#import numpy as np
import time
import pyautogui
#import uuid
#import win32api, win32con
#def click(x,y):
#    win32api.SetCursorPos((x,y))
#    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
#    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
#
#click(957, 317)


width, height = pyautogui.size()
#time.sleep(5)

# Pseudo "API" for that shitty KP software 
class KP_software_API:
    def __init__(self):
        self.button_locations = {}
        x = pyautogui.locateOnScreen('start.png')
        if not x:
            x = pyautogui.locateOnScreen('start_mouseover.png')
        if x:
            self.button_locations['wf_start'] = pyautogui.center(x)
        else:
            print('Could not find the "Start" button on the screen')
            raise Exception
            
        self.button_locations['wf_stop'] = self.button_locations['wf_start']
        self.button_locations['z_up'] = pyautogui.center(
                pyautogui.locateOnScreen('z_up.png'))
        self.button_locations['z_down'] = pyautogui.center(
                pyautogui.locateOnScreen('z_down.png'))
        zdown = self.button_locations['z_down']
        self.button_locations['tracking_show'] = (zdown[0] - 1206 + 1064,
                                                  zdown[1] - 171  + 404)
        self.button_locations['tracking_on'] = (zdown[0] - 1206 + 1062,
                                                  zdown[1] - 171  + 425)
        self.button_locations['z_step'] = (zdown[0] - 1206 + 1199,
                                           zdown[1] - 171  + 197)
    def button_click(self, button):
        pyautogui.click(self.button_locations[button])
        
    def set_z_step(self, step):
        pyautogui.click(self.button_locations['z_step'], clicks=2)
        pyautogui.typewrite('{0:d}'.format(step))
        pyautogui.typewrite(['enter'])
        
    def move_z_stage(self, step):
        if step != 0:
            self.set_z_step(abs(step))
            if step > 0:
                self.button_click('z_up')
            else:
                self.button_click('z_down')
        
    def start_wf_measurement(self):
#        x = pyautogui.locateOnScreen('start.png')
#        if not x:
#            x = pyautogui.locateOnScreen('start_mouseover.png')
#        if x:
##            print(list(x))
#            pyautogui.moveTo(pyautogui.center(x))
#            
            pyautogui.click(self.button_locations['wf_start'])
#            xx, yy = pyautogui.center(x)
#            click(xx, yy)
    def stop_wf_measurement(self):
        pyautogui.click(self.button_locations['wf_stop'])
        
    def save_measurement_file(self, filename):
        pathtofile = os.getcwd() + '\\' + filename
        pyautogui.typewrite(pathtofile)
        pyautogui.typewrite(['enter'])
            
#    def save_dat_file(self):
        
        
if __name__ == '__main__':
    print('OMG ITS MAIN!')
#    print(list(pyautogui.locateAllOnScreen('start.png')))
#    click(957,317)
    kapi = KP_software_API()
    time.sleep(1)
    kapi.move_z_stage(-10)
#    kapi.start_wf_measurement()
#    time.sleep(5)
#    uuid_here = uuid.uuid4().hex
#    filename = 'temp\\{0}.dat'.format(uuid_here)
#    kapi.save_measurement_file(filename)
#    data = np.genfromtxt(filename, skip_header=1, skip_footer=21, delimiter=',')
    
    
    
    

#pyautogui.click(pyautogui.center())

#pyautogui.moveTo(500, 500, duration=0.25)
#    for i in range(10):
#        print(pyautogui.position())
#        time.sleep(1)

#newpath = r'arbitrary' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)
    
#np.savetxt(newpath + '/' + 'file.txt', np.zeros(shape=(10)))

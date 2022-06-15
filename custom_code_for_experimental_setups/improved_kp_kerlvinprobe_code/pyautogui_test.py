# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
#import numpy as np
import time
import pyautogui
import numpy as np
#import win32api, win32con
#def click(x,y):
#    win32api.SetCursorPos((x,y))
#    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
#    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
#
#click(957, 317)


width, height = pyautogui.size()
#time.sleep(5)

# "API" for the shit 
class KP_software_API:
    def __init__(self):
        x = 1
        
    def start_wf_measurement(self):
        x = pyautogui.locateOnScreen('start.png')
        if not x:
            x = pyautogui.locateOnScreen('start_mouseover.png')
        if x:
#            print(list(x))
            pyautogui.moveTo(pyautogui.center(x))
            
            pyautogui.click(pyautogui.center(x), duration = 1)
#            xx, yy = pyautogui.center(x)
#            click(xx, yy)
            
#    def save_dat_file(self):
        
        
if __name__ == '__main__':
#    print(list(pyautogui.locateAllOnScreen('start.png')))
#    click(957,317)
    kapi = KP_software_API()
    kapi.start_wf_measurement()
    time.sleep(5)
    filename = 'test.dat'
    pathtofile = os.getcwd() + '\\' + filename
    pyautogui.typewrite(pathtofile)
    pyautogui.typewrite(['enter'])
#    data = np.genfromtxt(filename, skip_header=1, skip_footer=21, delimiter=',')
#    data = np.genfromtxt(filename, coun)
    

#pyautogui.click(pyautogui.center())

#pyautogui.moveTo(500, 500, duration=0.25)
#for i in range(10):
    
#    print(pyautogui.position())
#    time.sleep(1)

#newpath = r'arbitrary' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)
    
#np.savetxt(newpath + '/' + 'file.txt', np.zeros(shape=(10)))

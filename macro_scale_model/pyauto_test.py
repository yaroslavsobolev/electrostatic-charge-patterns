
import pyautogui
import time
import numpy as np

buttons = dict()
buttons['params'] = (1094, 272)
buttons['expression'] = (1321, 425)
buttons['expr_box'] = (1372, 740)
buttons['compute'] = (1581, 68)
buttons['blank'] = (1839, 948)

# def comsol_click(button_name):
#     time.sleep(0.1)
#     pyautogui.click(buttons[button_name])

# global_params_from_comsol = np.loadtxt('comsol/global_variable_probes.txt', skiprows=5)

x = pyautogui.locateOnScreen('puautogui_pics/parameters.png')

for i in range(100):
    print(pyautogui.position())
    time.sleep(0.5)
Software that controls the improvements made on top
of the KP scanning Kelvin probe system as described in the Supplementary
Information of the paper.

The improvements mainly compensate for the poor
choice of linear stages made by KP engineers.
The result is 3-100x faster scanning, 30-3000x more pixels.
Speedup is greater if the scan steps are larger, i.e. larger area is scanned.

Requires pyautogui package which is clicking the buttons on the native
KP software, since KP technology does not provide any API.

Requires thorlabs_apt package and thorlabs APT driver for controlling the
motors.

The main script here is `scan_large_area.py`

For settings in native software see screenshots `native-software-settings_1.PNG` 
and `native-software-settings_2.PNG`. The Z-tracking function of the KP Tech software must be disabled.

Before running the Python script, you should:
* Align the sample to the scanning plane with the two-axis goniometer. Use Thorlabs native "APT User" software for that.
  After your alignment, the "gradient" value in the KP Software must be roughtly 50-60 everywhere across the sample area.
* Close the Thorlabs native "APT User" software.
* Switch to the "Work function measurement" tab in the KP Technology native software (see `native-software-settings_2.PNG`).
The "Start" button (`start.png`) must be visible on your computer screen, since it will be searched and clicked by `PyAutoGui`.
The Z-tracking function of KP Tech software must be disabled. 
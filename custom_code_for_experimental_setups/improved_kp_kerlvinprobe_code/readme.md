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
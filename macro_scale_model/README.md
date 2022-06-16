# Numerical simulation of ESDs that accompany the stamp detachment process.

The `macro-scale.py` script outsources the electrostatic FEM calculations to COMSOL Multiphysics.
Since we did not have license for full API of COMSOL, the interaction of Python with COMSOL is 
implemented via pyautogui: Python clicks certain buttons and changes parameters in an open COMSOL window,
clicks "Solve" button, then COMSOL finished calculation and saves results to files in disk, and Python reads
data from these files. This happens in every iteration of the simulation algorithm described in the paper.

If you have a full API licence, then use the API. If you have to resort to the same pyautogui hack as here, 
do the following:

* You should printscreen the images of respective elements of COMSOL interface and save them to `pyautogui_pics` folder.
See current pictures there for reference.
* You should find out the screen locations of certain buttons on the COMSOL window on your computer screen 
  (use `pyauto_test.py` helper script for that purpose) and write them to `buttons` dictionary in the 
  `macro-scale.py` script (lines 32-37).
* COMSOL must be configured to automatically save the calculation results to certain `.txt` files in `comsol/` folder,
as shown in the screenshots `comsol/comsol_settings_1.png` and `comsol/comsol_settings_2.png`.


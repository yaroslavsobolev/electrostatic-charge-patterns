# this function converts the analog (Anymetre) hygrometer data
# into true humidity
import numpy as np

def correct_humidity(analog, calibration_folder, digital=False):
    # load calibration parameters
    switch_location = np.load(calibration_folder + 'calibration_switch.npy')
    fit_function = np.load(calibration_folder + 'calibration_line.npy')
    if analog < switch_location:
        result = analog
    else:
        if digital:
            result = digital
        else:
            result = np.poly1d(fit_function)(analog)
    return result
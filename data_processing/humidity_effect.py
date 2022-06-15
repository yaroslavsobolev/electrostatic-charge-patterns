import numpy as np
import matplotlib.pyplot as plt
from hygrometer_calibration.calibration_function import correct_humidity

for x in [72, 70, 41, 62, 50]:
    print('for {0} the corrected value is {1:.2f}'.format(x,
                                                          correct_humidity(x, calibration_folder='hygrometer_calibration/')))

corrected = [correct_humidity(x, calibration_folder='hygrometer_calibration/') for x in [72, 70, 41, 62, 50]]
print(corrected)

for x in range(10, 100, 10):
    print('for {0} the corrected value is {1:.2f}'.format(x,
                                                          correct_humidity(x, calibration_folder='hygrometer_calibration/')))

# # now import pylustrator
# import pylustrator
#
# # activate pylustrator
# pylustrator.start()

file_list = [
'E:\\Lab\\improved_kp_kelvinprobe\\20191015_5cm_3in_62RH_eq30min_oldPDMS5to1_PMMAtol_uniformspeeds_0p1_B01_copy',

'F:\\PDMS-PMMA_delamination_experiments\\kelvin_probe\\'
'20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_A01',

'E:\\Lab\\improved_kp_kelvinprobe\\20191030_5cm_3in_50RH_ambRH38_eq30min_newPDMS5to1_PMMAtol_uniformspeeds_0p5_C01_copy',

'E:\\Lab\\improved_kp_kelvinprobe\\20191107_5cm_3in_41RH_ambRH36_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_B01',

'E:\\Lab\\improved_kp_kelvinprobe\\20191120_5cm_3in_72RH_ambRH21_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p5_D01_2',

'E:\\Lab\\improved_kp_kelvinprobe\\20191123_5cm_3in_70RH_ambRH44_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_dark_0p4_B01',

'E:\\Lab\\improved_kp_kelvinprobe\\20191113_70RH_B01',

'E:\\Lab\\improved_kp_kelvinprobe\\20191118_77RH_B01',

'E:\\Lab\\improved_kp_kelvinprobe\\20191108_5cm_3in_31RH_ambRH32_eq30min_newPDMS5to1_PMMAtol_newwire_uniformspeeds_slowstart_0p3_E01'
]
data = []
rh_analog = [62, 60, 50, 41, 72, 70, 70, 77, 31]
rh_true = [correct_humidity(x, calibration_folder='hygrometer_calibration/') for x in rh_analog]
for folder in file_list:
    filename = folder + '\\' + 'max_density_metrics.txt'
    data_here = np.loadtxt(filename, delimiter='\t', skiprows=1)
    data.append(data_here)
        # data = np.append(data_here, data, axis=1)
data = np.vstack(data)
for_saving = np.vstack((np.array(rh_true), data[:, 0], data[:, 1])).T
np.save('histogram_metrics_vs_humidity', for_saving)
plt.plot(rh_true, data[:,0], 'o', label='(SKP) Positive peak on histogram of charge density', color='C1')
plt.plot(rh_true, data[:,1], 'o', label='(SKP) Maximum value of positive charge density', color='C2')
plt.xlabel('Relative humidity, %')
plt.ylabel('Estimated initial charge density, nC$\cdot$cm$^{-2}$')
plt.legend()
plt.show()
print(1)
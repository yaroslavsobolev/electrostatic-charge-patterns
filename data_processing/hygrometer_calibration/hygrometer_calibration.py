import numpy as np
import matplotlib.pyplot as plt
import glob

data_folder = 'experimental_data/PDMS-PMMA_delamination_experiments/data/'

folders = [f for f in glob.glob(data_folder + "**/", recursive=False)]

readings = []
for f in folders:
    if '2hygrometer' in f:
        pieces = f.split('_')
        print(pieces)
        hygrometer_location = pieces.index('2hygrometer')
        analog_rh = int(pieces[hygrometer_location - 1][:-2])
        digital_rh = int(pieces[hygrometer_location + 1][:-2])
        readings.append([analog_rh, digital_rh])

readings_list = np.array(readings)
readings_for_linear_fit = np.array([r for r in readings if r[0] > 51])
fit_function = np.polyfit(readings_for_linear_fit[:,0], readings_for_linear_fit[:,1], 1)
print(fit_function)
fig, ax1 = plt.subplots()

color = 'C0'
ax1.set_xlabel('Analog hygrometer (Anymetre) reading, %RH')
ax1.set_ylabel('Digital hygrometer (Testo) reading, %RH', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.scatter(readings_list[:,0], readings_list[:,1], s=20, alpha=0.3)
ax1.errorbar(x=readings_list[:,0], y=readings_list[:,1], xerr=5, yerr=3, linestyle='', capsize=3)

ax1.set_ylim(0, 100)
ax1.set_xlim(0, 100)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylim(0, 100)
color = 'C1'
ax2.set_ylabel('Corrected (true) relative humidity, %RH', color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)
swtich_location = 40.5
xs = np.linspace(0, swtich_location, 100)
ax2.plot(xs, xs, color='C1')
xs = np.linspace(swtich_location, 88, 100)
ys = np.poly1d(fit_function)(xs)
ax2.plot(xs, ys, color='C1')
ax1.axvline(x=swtich_location, linestyle='--', color='black', alpha=0.5)
# fig.savefig('hygrometer_calibration.png', dpi=300)
# np.save('calibration_switch', swtich_location)
# np.save('calibration_line', fit_function)


plt.show()
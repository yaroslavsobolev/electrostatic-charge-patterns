import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import psds
import imageio
import pylab

base_folder = 'D:/Docs/Science/UNIST/Projects/Vitektrification/pattern_scaling/'

data = np.load(base_folder + 'macro_scale_model/results/test1_pcd9_hyst3o8_zeroend_12_18_2019-16_59_10__good_run/datasets/coverage_0.060.npy')
# data = np.load('E:\\Yaroslav\\test1_pcd15_hyst3o8_zeroend_12_26_2019-12_35_18\\datasets\\coverage_0.030.npy')
fs = data.shape[0]/10
f, Pxx_den = signal.welch(data, fs)


data = np.load(base_folder + 'macro_scale_model/results/test1_pcd8_hyst3o8_zeroend_12_19_2019-12_54_13__good_run/datasets/coverage_0.030.npy')
fs2 = data.shape[0]/10
f2, Pxx_den2 = signal.welch(data, fs2)

data_for_export = np.vstack((f2, Pxx_den2+Pxx_den))
# np.save('data_processing/power_spectra/theoretical_model/000', data_for_export)

plt.loglog(f2, Pxx_den2+Pxx_den)
# plt.ylim([0.5e-3, 1])
plt.xlim(0.1,10)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.grid(True,which="major",ls="-")
plt.show()

#
# ps = np.abs(np.fft.fft(data))**2
#
# time_step = 10/data.shape[0]
# freqs = np.fft.fftfreq(data.size, time_step)
# idx = np.argsort(freqs)
#
# plt.loglog(abs(freqs[idx]), ps[idx])
# plt.show()

# im0 = imageio.imread('D:\\Docs\\Science\\UNIST\\Projects\\Vitektrification\\paper_at_all_scales\\pics\\input\\for_power_spectra\\output7.tif')
pix_per_mm = 10/np.sqrt(2)
im = np.stack((data for i in range(data.shape[0])))
plt.imshow(im)
hanning = True
freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
data_for_export = np.vstack((freqG * pix_per_mm, psdG))
pylab.figure(2)
pylab.clf()
pylab.loglog(freqG*pix_per_mm,psdG,label='Power spectrum')
pylab.loglog(f2, 1.22e-6/2.35e-10*Pxx_den2, label='1d')
pylab.loglog(f2, 1.22e-6/2.35e-10*8.3/1.8*Pxx_den2/f2, label='1d vid by f')
xs = np.linspace(0.1, 10, 100)
# ys = 100*xs**(-2)
pylab.loglog(f2, 1/8e4*f2**(-2), label='-2', color='black')
pylab.legend(loc='best')
pylab.xlabel("Spatial frequency ($mm^{-1}$)")
pylab.ylabel("Normalized Power")
pylab.grid(True,which="both",ls="-")
plt.legend()
plt.show()
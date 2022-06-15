import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab

im0 = imageio.imread('Literature/Bertein_1973.PNG')
pix_per_mm = 75.8/12
im = np.sum(im0, axis=2)
plt.imshow(im)
hanning = True
freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
area = im.shape[0] * im.shape[1] / (pix_per_mm) ** 2
data_for_export = np.vstack((freqG * (pix_per_mm / np.sqrt(2)), psdG / area))
np.save('Literature/Bertein', data_for_export)
pylab.figure(2)
pylab.clf()
pylab.loglog(freqG * (pix_per_mm / np.sqrt(2)), psdG / area,label='Power spectrum')
pylab.legend(loc='best')
pylab.xlabel("Spatial frequency ($mm^{-1}$)")
pylab.ylabel("Normalized Power")
pylab.grid(True,which="both",ls="-")
plt.legend()
plt.show()

im0 = imageio.imread('Literature/Hull_1949.PNG')
pix_per_mm = 1053/67
im = np.sum(im0, axis=2)
plt.imshow(im)
hanning = True
freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
# data_for_export = np.vstack((freqG * pix_per_mm, psdG))
area = im.shape[0] * im.shape[1] / (pix_per_mm) ** 2
data_for_export = np.vstack((freqG * (pix_per_mm / np.sqrt(2)), psdG / area))
np.save('Literature/Hull_1949', data_for_export)
pylab.figure(2)
pylab.clf()
pylab.loglog(freqG * (pix_per_mm / np.sqrt(2)), psdG / area,label='Power spectrum')
pylab.legend(loc='best')
pylab.xlabel("Spatial frequency ($mm^{-1}$)")
pylab.ylabel("Normalized Power")
pylab.grid(True,which="both",ls="-")
plt.legend()
plt.show()
import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab

im0 = imageio.imread('Literature/Bertein_1973.PNG')
pix_per_mm = 75.8/12
im1 = np.mean(im0, axis=2)[600:1000,:]
plt.imshow(im1)
hanning = True
pylab.figure(2)
pylab.clf()

names = ['quarter_width', 'half_width', 'full_width', 'full_height']
for i, im in enumerate([im1[:, :250], im1[:, :500], im1[:, :], np.mean(im0, axis=2)]):
    freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
    area = im.shape[0] * im.shape[1] / (pix_per_mm) ** 2
    pylab.loglog(freqG * (pix_per_mm / np.sqrt(2)), psdG / area * (2 * np.pi * freqG * (pix_per_mm / np.sqrt(2))) ,label=names[i], alpha=0.5)
    imageio.imwrite('Literature/temp/{0}.png'.format(names[i]), im)

pylab.legend(loc='best')
pylab.xlabel("Spatial frequency ($mm^{-1}$)")
pylab.ylabel("Power spectral density")
pylab.grid(True,which="both",ls="-")
plt.legend()
plt.show()
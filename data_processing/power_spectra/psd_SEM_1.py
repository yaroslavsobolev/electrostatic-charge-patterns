import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab
from skimage.transform import resize

im0 = imageio.imread('Literature/mountain_test.png')
mindim = np.min(im0.shape[0:1])
im = im0[:mindim, :mindim, 0]
# for i in range(mindim):
#     for j in range(mindim):
#         im[i, j] =
# im = np.random.normal(100, 30, im.shape)
line = 100+50*np.sin(2*np.pi/20*np.arange(mindim, dtype=np.float))
# line = np.random.normal(100, 30, mindim)
for i in range(mindim):
        im[i, :] = line
pix_per_mm = 1/np.sqrt(2)
# im = im0[:,:]
plt.imshow(im)
hanning = True

# xxx = psds.power_spectrum(im, oned=True, hanning=hanning)
freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
area = im.shape[0]*im.shape[1]/(pix_per_mm)**2
# data_for_export = np.vstack((freqG * pix_per_mm, psdG))
# np.save('SEM/001', data_for_export)
pylab.figure(2)
pylab.clf()
pylab.loglog(freqG*pix_per_mm,psdG/area,label='Power spectrum')

# plt.show()
# pylab.loglog(freqG*pix_per_mm,psdG*freqG,label='Power spectrum, wavenum_true_direct')

# freqG,psdG = psds.power_spectrum(im, oned=True, hanning=hanning, wavnum_scale=True)
# pylab.loglog(freqG*pix_per_mm,psdG,label='Power spectrum, wavenum_true')
# im = resize(im, (im.shape[0]/2, im.shape[1]/2), preserve_range=True)
# cut = int(round(im.shape[0]/2))
cut = 800
factor = im.shape[0]/(im.shape[0] - cut)
im = im[cut:, :]
# im = np.concatenate((im, im), axis=0)
area = im.shape[0]*im.shape[1]/(pix_per_mm)**2
# plt.figure(4)
# plt.imshow(im)
pix_per_mm = 1/np.sqrt(2)
# plt.figure(5)
freqG1,psdG1 = psds.power_spectrum(im, oned=True, hanning=hanning)
pylab.loglog(freqG1*pix_per_mm,psdG1/area,label='Power spectrum 2x')

pylab.legend(loc='best')
pylab.xlabel("Spatial frequency ($mm^{-1}$)")
pylab.ylabel("Normalized Power")
pylab.grid(True,which="both",ls="-")
plt.legend()

f3 = plt.figure(3)
plt.psd(im[0,:])
plt.show()
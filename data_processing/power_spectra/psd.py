import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab

# im0 = imageio.imread('spdtest.png')
# im = im0[:,:]

# im0 = imageio.imread('spdtest.jpg')
# im = np.sum(im0, axis=2)

im0 = imageio.imread('Literature\\Baytekin\\Raman_minus.PNG')
im = np.sum(im0, axis=2)

plt.imshow(im0)

# xx = psds.PSD2(im)
# plt.imshow(np.log(xx))
# plt.colorbar()
# plt.show()

freqG,psdG = psds.power_spectrum(im, oned=True)
print(freqG[0])
print(1/im.shape[0])

im0 = imageio.imread('Literature\\Baytekin\\Raman_plus.PNG')
im = np.sum(im0, axis=2)

freqGB,psdGB = psds.power_spectrum(im, oned=True)
print(1)
pylab.figure(2)
pylab.clf()
pylab.loglog(freqG,psdG,label='Power spectrum, minus')
pylab.loglog(freqGB,psdGB,label='Power spectrum, plus')
pylab.legend(loc='best')
# pylab.axis([7,400,1e-7,1])
pylab.xlabel("Spatial frequency ($pixels^{-1}$)")
pylab.ylabel("Normalized Power")
pylab.grid(True,which="both",ls="-")
# pylab.savefig("example_psd_scale.png")


pylab.savefig("example_psd_scale.png")
plt.show()
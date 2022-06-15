import psds
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pylab

def get_PSD_from_SEM_file(target_file, saveto, pix_per_mm=False, do_plot=False):
    im0 = imageio.imread(target_file)
    im = im0[:, :]
    plt.imshow(im0)
    if pix_per_mm==False:
        pix_per_mm = np.max(im.shape)/10
        print('evaluating pix per mm: {0}'.format(pix_per_mm))
    hanning = True
    freqG, psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
    area = im.shape[0] * im.shape[1] / (pix_per_mm) ** 2
    data_for_export = np.vstack((freqG * (pix_per_mm/np.sqrt(2)), psdG/area))
    np.save(saveto, data_for_export)
    if do_plot:
        pylab.figure(2)
        pylab.clf()
        pylab.loglog(freqG * (pix_per_mm/np.sqrt(2)), psdG/area, label='Power spectrum')
        pylab.legend(loc='best')
        pylab.xlabel("Spatial frequency ($mm^{-1}$)")
        pylab.ylabel("Normalized Power")
        pylab.grid(True, which="both", ls="-")
        plt.legend()
        plt.show()

# get_PSD_from_SEM_file(target_file='D:\\Docs\\Science\\UNIST\\Projects\\Vitektrification\\'
#                                   'paper_at_all_scales\\pics\\input\\for_power_spectra\\output7.tif',
#                       saveto='SEM/008',
#                       pix_per_mm = 542.8,
#                       do_plot=True)
SEM_data_folder = 'experimental_data/SEM/'
filenames = [
SEM_data_folder + 'output1.tif',
SEM_data_folder + '20180719_content_aware_fill.png',
SEM_data_folder + 'output5.tif',
SEM_data_folder + 'output7.tif'
]
saveto_ids = [1,3,6,8]
pix_per_mm = [False, 5344/10, False, False]
for i,filename in enumerate(filenames):
    print(filename)
    get_PSD_from_SEM_file(target_file=filename,
                          saveto='SEM/{0:03d}'.format(saveto_ids[i]),
                          pix_per_mm=pix_per_mm[i],
                          do_plot=False)

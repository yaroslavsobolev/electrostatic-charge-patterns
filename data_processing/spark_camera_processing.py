import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter

target_folder_base = 'F:/PDMS-PMMA_delamination_experiments/20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire' \
                '_uniformspeeds_slowstart_dark_0p4_A01/'
target_folder = target_folder_base + 'video/frames/'

spark_sequence = [5242, 5524, 5100 + 593, 5100 + 933, 5100 + 1104, 5100 + 1111, 5100 + 1156]

def get_filename(n):
    return target_folder + 'frame_X{0:d}.tif'.format(n)

f1, ax1 = plt.subplots()
f2, ax2 = plt.subplots()
for_stack = []
for i, s in enumerate(spark_sequence[:-1]):
    image1 = ndimage.imread(get_filename(s))
    im_med = ndimage.median_filter(image1, 5)
    im_blur = ndimage.gaussian_filter(im_med, 0.7)
    fst = np.copy(im_blur)
    vmin = 525
    fst[fst < vmin] = vmin
    for_stack.append(np.copy(fst))
    ax1.imshow(image1, cmap='gray', vmin=495, vmax = 600)
    f1.savefig(target_folder_base + 'processing/sparks_video/individual_sparks/raw/{0:d}'.format(i), dpi=600)
    ax2.imshow(im_blur, cmap='magma', vmin = 495, vmax = 600) #plt.cm.gray
    f2.savefig(target_folder_base + 'processing/sparks_video/individual_sparks/processed/{0:d}'.format(i), dpi=600)

sum_img = np.zeros_like(for_stack[0])
for fst1 in for_stack:
    sum_img += fst1
f3 = plt.figure(3)
plt.imshow(sum_img, cmap='magma')
# plt.show()
# coll_list = []
# for N in range(5100, 6400, 1):
#     image1 = ndimage.imread(get_filename(N))
#     collapse = np.mean(image1, axis=1)
#     coll_list.append(savgol_filter(collapse, 13, 3))
# trace = np.vstack(tuple(coll_list))
# np.save('trace2', trace)
# # trace = np.load('trace1.npy')
# plt.imshow(trace, cmap='inferno')
# plt.axes().set_aspect(aspect="auto")


# f3.savefig(target_folder_base + 'processing/sparks_video/combined_sparks.png'.format(i), dpi=600)
plt.show()
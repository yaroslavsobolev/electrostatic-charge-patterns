import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter

target_folder_base = 'E:/PDMS-PMMA_delamination_experiments/20191125_5cm_3in_60RH_ambRH43_eq30min_newPDMS5to1_PMMAtol_newwire' \
                '_uniformspeeds_slowstart_dark_0p4_A01/'
target_folder = target_folder_base + 'video/frames/'

# unseen previous sparks are 4643, 4815,
spark_sequence = [5242, 5524, 5100 + 593, 5100 + 933, 5100 + 1104, 5100 + 1111, 5100 + 1156]

def get_filename(n):
    return target_folder + 'frame_X{0:d}.tif'.format(n)

# f1, ax1 = plt.subplots()
# f2, ax2 = plt.subplots()
# fig, axarr = plt.subplots(5, figsize=(6,8))

gs_top = plt.GridSpec(5, 1, top=0.95)
gs_base = plt.GridSpec(5, 1, hspace=0.1)

fig= plt.figure(figsize=(6,8))
axarr = []
axarr.append(fig.add_subplot(gs_top[0,:]))
axarr.append(fig.add_subplot(gs_base[1,:]))
axarr.append(fig.add_subplot(gs_base[2,:]))
axarr.append(fig.add_subplot(gs_base[3,:]))
axarr.append(fig.add_subplot(gs_base[4,:]))


kp_image = ndimage.imread('spark_video/bipolar.png')
axarr[3].imshow(np.transpose(kp_image[985:3579,3385:3950,:], (1,0,2)))
axarr[3].set_axis_off()
axarr[1].set_axis_off()
axarr[2].set_axis_off()
axarr[4].set_axis_off()
axarr[0].set_xlabel('Time, s')
axarr[0].margins(0.05)
axarr[0].set_ylabel('Electric current, nA')
# plt.show()
peaks = np.load(target_folder_base+'electrometer/peaks.npy')
t = np.load(target_folder_base+'electrometer/time.npy')
t_for_graph = np.load(target_folder_base+'electrometer/time_for_graph.npy')
dary = np.load(target_folder_base+'electrometer/discharge_magnitude_graph.npy')
delamination_start_index = int(np.load(target_folder_base+'electrometer/delamination_start_index.npy'))
steplen = int(np.load(target_folder_base+'electrometer/steplen.npy'))
t_0_for_graph = 50
for_stack = []

def camera_frame_to_electrometer_frame(x):
    return int(round(steplen + delamination_start_index + 2186 + (x - 5242)/(5524 - 5242)*(2633-2186)))


frame_range = list(range(4050, 5100 + 1200))
for i, s in enumerate(frame_range[:]):
# for i, s in enumerate(spark_sequence[:]):
    for axx in axarr[:3]:
        axx.clear()
    axarr[0].set_xlabel('Time, s')
    axarr[0].margins(0.05)
    axarr[0].set_ylabel('Electric current, nA')
    axarr[1].set_axis_off()
    # axarr[1].set_title('Raw footage, top view')
    axarr[2].set_axis_off()
    axarr[0].plot(t_for_graph - t_0_for_graph,
             dary/0.00228,
             color='C0', linewidth=0.9)
    for step_loc in peaks:
        time_here = t[steplen + delamination_start_index + step_loc]
        axarr[0].axvline(x=time_here - t_0_for_graph, color='grey', alpha=0.5)

    elec_lim = camera_frame_to_electrometer_frame(s)

    time_here = t[elec_lim]
    axarr[0].axvline(x=time_here - t_0_for_graph, color='red', alpha=0.4)

    axarr[0].set_title('Time: {0:.3f} s'.format(t[elec_lim]-t_0_for_graph))

    image1 = ndimage.imread(get_filename(s))
    im_med = ndimage.median_filter(image1, 5)
    im_blur = ndimage.gaussian_filter(im_med, 0.7)
    # fst = np.copy(im_blur)
    vmin = 525
    # fst[fst < vmin] = vmin
    # for_stack.append(np.copy(fst))
    axarr[1].imshow(np.fliplr(image1.T), cmap='gray', vmin=495, vmax = 600)
    axarr[1].set_aspect(aspect="auto")
    axarr[1].set_xlim(-9, 175)
    axarr[1].set_ylim(0, 45)
    vmax_here = np.max(im_blur)
    if vmax_here < 580:
        vmax_here = 580
    axarr[2].imshow(np.fliplr(im_blur.T), cmap='magma', vmin = 500, vmax = vmax_here)
    axarr[2].set_aspect(aspect="auto")
    axarr[2].set_xlim(-9, 175)
    axarr[2].set_ylim(0, 45)
    fig.savefig('F:/PDMS-PMMA_delamination_experiments/sparks/frames/{0:08d}.tif'.format(i), dpi=100)
    # f1.savefig(target_folder_base + 'processing/sparks_video/individual_sparks/raw/{0:d}'.format(i), dpi=600)
    # ax2.imshow(im_blur, cmap='magma', vmin = 495) #plt.cm.gray
    # f2.savefig(target_folder_base + 'processing/sparks_video/individual_sparks/processed/{0:d}'.format(i), dpi=600)
    # plt.show()
# sum_img = np.zeros_like(for_stack[0])
# for fst1 in for_stack:
#     sum_img += fst1
# f3 = plt.figure(3)
# plt.imshow(sum_img, cmap='magma')
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
# plt.show()
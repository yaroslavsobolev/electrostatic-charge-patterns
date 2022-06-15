# -*- coding: utf-8 -*-
"""
Image processing of the motion of detachment front during the delamination
of PDMS stamp from a thin layer of polymer on a substrate.

It also uses data from electrometer connected to the conductive substrate
that measures the charge on this terminal (substrate) synchronously with
camera recording.

Created on Tue Aug 22 01:28:37 2017

@author: Yaroslav I. Sobolev
"""

import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy import interpolate
from scipy import signal
from skimage.morphology import skeletonize
from skimage import io
from skimage.color import rgb2hsv
from skimage.feature import register_translation
from skimage.transform import AffineTransform, warp, resize
from skimage import feature
from skimage.filters import roberts, sobel
import pickle
import time
t0 = time.time()
from lmfit.models import GaussianModel, ConstantModel

def shift_image(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)
    return shifted

frames_directory = 'E:\\PDMS-PMMA_delamination_experiments\\data\\20191017_5cm_3in_62RH_eq30min_oldUVPDMS_PMMAtol_uniformspeeds_0p1_B01\\video'
istart = 1621+25 #int(round(57*30))+5+20
iend = 2280  #2280
c = 0.0033
deriv_step = 2
testing = True

import pylab
import imageio
filename = frames_directory + '\\video.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

# find a time where the frame changes the most -- this will be the moment when the stamp falls off
previous_frame = vid.get_data(2200)
indices = []
diffs_of_frames = []
for i, im in enumerate(vid):
    if i < 2201:
        continue
    print(i)
    diff = np.sum(np.abs(im - previous_frame))
    diffs_of_frames.append(diff)
    indices.append(i)
    previous_frame = np.copy(im)
plt.plot(indices, diffs_of_frames)
print('Falling moment is frame {0}'.format(indices[np.argmax(diffs_of_frames)]))
plt.show()

def stabilize_video(patch_coordinates, testing=False):
    xmin, xmax, ymin, ymax = patch_coordinates
    frames = []
    shift_magnitudes = []
    for i, im in enumerate(vid):
        if (not testing) and (i < istart):
            continue#enumerate([vid.get_data(26*30), vid.get_data(istart), vid.get_data(istart+1)]):#
        # print('Mean of frame %i is %1.1f' % (i, im.mean()))
        frame = sk.img_as_float(np.sum(im, axis=2))
        if testing:
            plt.imshow(frame)
            plt.show()
        patch_for_alignment = sobel(frame[xmin:xmax, ymin:ymax])
        if testing:
            f1 = plt.figure(2)
            plt.imshow(patch_for_alignment)
            plt.show()
        if i == istart:
            ref_patch = np.copy(patch_for_alignment)
            if testing:
                f1.savefig('refpatch.png')
            continue
        shift, error, diffphase = register_translation(src_image=ref_patch, target_image=patch_for_alignment, upsample_factor = 20)
        if testing:
            print('Shift={0}'.format(shift))
            f1 = plt.figure(1)
            plt.imshow(patch_for_alignment)
            f1.savefig('before_alignment.png')
            aligned_patch = shift_image(patch_for_alignment, vector=[-1*shift[1], -1*shift[0]])
            plt.imshow(aligned_patch)
            f1.savefig('after_alignment.png')
        aligned_image = shift_image(frame, vector=[-1 * shift[1], -1 * shift[0]])
        frames.append(np.copy(aligned_image))
        shift_magnitudes.append(np.linalg.norm(shift))
        print(i)
    all_frames = np.stack(frames)
    return all_frames, shift_magnitudes

# frames, shift_magnitudes = stabilize_video([471, 520, 3, 58])
# np.save(frames_directory + '\\video_stabilized_float', frames)
# np.save(frames_directory + '\\first_frame', istart+1)
# plt.plot(shift_magnitudes)
# plt.show()
def load_stabilized_frames(frames_directory):
    print('Loading stabilized frames...')
    frames = np.load(frames_directory + '\\video_stabilized_float.npy')
    first_frame_number = np.load(frames_directory + '\\first_frame.npy')
    print('...Loaded.')
    return frames, first_frame_number

frames, first_frame_number = load_stabilized_frames(frames_directory)

# plt.imshow(frames[0])
# plt.show()

# nums = [10, 287]
# for num in nums:
#     image = vid.get_data(num)
#     fig = pylab.figure()
#     fig.suptitle('image #{}'.format(num), fontsize=20)
#     pylab.imshow(image)
# pylab.show()

def get_frame(i, directly_from_video=False):
    if directly_from_video:
        return sk.img_as_float(np.sum(vid.get_data(i), axis=2))
    else:
        return frames[i-first_frame_number,:,:]


# ================================OLD STUFF THAT TAKES A LOT OF MEMORY==========
# data = []
# for k,i in enumerate(range(istart, iend)):
#     filename = '{0:07d}.tif'.format(i)
#     pic = sk.img_as_float(io.imread(frames_directory + filename))
#     data.append(pic)
#
# deriv = []
# for k in range(len(data)-4*deriv_step):
#     # Numerical first derivative using five-poinc stencil
#     deriv.append((-1*data[k+4*deriv_step] + 8*data[k+3*deriv_step] - 8*data[k+1*deriv_step] + data[k])/12/deriv_step)
# data = []
# ==============================================================================

def get_deriv_image(nframe, deriv_step, do_show=True):
    local_data = []
    for i in [nframe - 2 * deriv_step, nframe - 1 * deriv_step, nframe + 1 * deriv_step, nframe + 2 * deriv_step]:
        # frame = #vid.get_data(num)
        # frame_here = get_frame(i)
        pic = get_frame(i)
        # pic = sk.img_as_float(frame_here[:,:,0])
        # pic = sk.img_as_float(rgb2hsv(frame_here)[:,:,0])
        # plt.imshow(pic)
        # plt.show()
        local_data.append(pic)

    deriv_pic = (-1 * local_data[3] + 8 * local_data[2] - 8 * local_data[1] + local_data[0]) / 12 / deriv_step
    # if do_show:
        # print(np.mean(deriv_pic))
        # print(np.max(deriv_pic))
        # print(np.max(deriv_pic)-np.mean(deriv_pic))
        # print((np.max(deriv_pic) - np.mean(deriv_pic))/np.mean(deriv_pic))
        # plt.imshow(deriv_pic)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(np.abs(deriv_pic))
        # plt.colorbar()
        # plt.show()
    return deriv_pic


# plt.imshow(deriv[40], cmap = 'Greys_r', origin = 'lower')
# plt.autoscale(False)
# fig = plt.figure()
# if not testing:
#     ax = fig.add_subplot(111)
#     ax.set_aspect(1)

def find_fronts_on_frames():
    front_curves = []
    fig_fronts, ax_fronts = plt.subplots()
    fig_stripes, ax_stripes_arr = plt.subplots(3, sharex=True, figsize=(4,6))
    fig_diffs, ax_diffs = plt.subplots()
    for k in range(istart, iend):
        print('Frame is {0}'.format(k))
        ## Skip all frames except every 10th
        # if k % 10 > 0:
        #     continue
        # if k > 29383:
        #     deriv_step = min(50, int(round(10 + (k - 29383) / 20)))
        #     c = 0.0033 * 10 / deriv_step

        #    print('Frame {0}'.format(k))
        # k = 40
        # img = deriv[k]
        #    c = 0.042# + k/1000*0.02
        roi = [120, 302, 420, 1100]
        img = get_deriv_image(k, deriv_step)[roi[0]:roi[1],roi[2]:roi[3]]
        raw_img = get_frame(k)[roi[0]:roi[1],roi[2]:roi[3]]
        # ax_diffs.clear()
        maxabs = np.max(np.abs(img))
        # ax_diffs.imshow(img, vmin=-1*maxabs, vmax=maxabs, cmap='seismic')
        # fig_diffs.savefig(frames_directory + '\\delamination_fronts\\diffs\\{0:08d}.png'.format(k), dpi=60)
        # plt.show()
        def get_front_location_in_stript(x_stripe, filename_for_stripe, upsampling_factor=10):
            def find_peak_with_upsampling(stripe, upsampling_factor = 10):
                interpolator_here = interpolate.interp1d(np.linspace(0, stripe.shape[0]-1, stripe.shape[0]), stripe,
                                                         kind='cubic', fill_value='extrapolate')
                # nsteps_upsampled = 1 + (stripe.shape[0]-1)*(upsampling_factor-1)
                xnew = np.arange(0, stripe.shape[0], step=1/upsampling_factor)#stripe.shape[0]*upsampling_factor,retstep=True)
                ynew = interpolator_here(xnew)
                peaks, _ = signal.find_peaks(ynew, prominence=0.2e-9, width = [1*upsampling_factor, 40*upsampling_factor]) #
                if peaks.shape[0] == 0:
                    main_peak = -1
                    print('line {0} -> No peak found.'.format(x_stripe))
                else:
                    main_peak = peaks[np.argmax(_['prominences'])]
                main_peak /= upsampling_factor
                return main_peak, xnew, ynew, peaks
            # stripe = np.abs(img[x_stripe,:])
            stripe = signal.savgol_filter(np.abs(img[x_stripe,:]), window_length=27, polyorder=3)
            main_peak_in_time_derivative_stripe, xnew, ynew, peaks = find_peak_with_upsampling(stripe)
            search_zone = 30
            zone_borders = [int(round(main_peak_in_time_derivative_stripe - search_zone)),
                            int(round(main_peak_in_time_derivative_stripe + search_zone))]
            stripe2 = np.abs(np.diff(raw_img[x_stripe, :]))[zone_borders[0]:zone_borders[1]]
            if main_peak_in_time_derivative_stripe >= 0 and stripe2.shape[0] > 20:

                # # GAUSSIAN FITTING VERSION. IT IS SLOW.
                # gmodel = GaussianModel() + ConstantModel()
                # try:
                #     stripe2 = signal.savgol_filter(stripe2, window_length=min(5, stripe2.shape[0]), polyorder=1)
                # except ValueError:
                #     print('Window too small for smoothing')
                #     pass
                #
                # params = gmodel.make_params(amplitude=np.max(stripe2),
                #                             center=search_zone,
                #                             sigma=15,
                #                             c=np.min(stripe2))
                # gmodel.set_param_hint('amplitude', min=0)
                # gmodel.set_param_hint('sigma', max=40)
                # fit_result = gmodel.fit(stripe2, params, x=np.arange(stripe2.shape[0]))
                #
                # # print(fit_result.fit_report())
                # main_peak = fit_result.best_values['center'] + 0.5 + zone_borders[0]
                # # plt.plot(xdata, fit_result.best_fit, 'r-')
                #

                # PEAK FINDING VERSION
                stripe2 = signal.savgol_filter(stripe2, window_length=17, polyorder=3)
                main_peak_in_space_diff_stripe, xnew2, ynew2, peaks2 = find_peak_with_upsampling(stripe2)
                main_peak = main_peak_in_space_diff_stripe + 0.5 + zone_borders[0]
                time_deriv_only = False
            else:
                time_deriv_only = True
                main_peak = main_peak_in_time_derivative_stripe

            # ## Implementation without upsampling
            # peaks, _ = signal.find_peaks(stripe, prominence=2e-9, width = [1, 30]) #
            # if peaks.shape[0] == 0:
            #     main_peak = -1
            #     print('line {0} -> No peak found.'.format(x_stripe))
            # else:
            #     main_peak = peaks[np.argmax(_['prominences'])]

            # if main_peak_in_time_derivative_stripe >= 0 and testing and x_stripe == 100:
            #     # print('Max prominence = {0}'.format(np.max(_['prominences'])))
            #     for axarr in ax_stripes_arr:
            #         axarr.clear()
            #     # ax_stripes_arr[0].clear()
            #     ax_stripes_arr[0].plot(xnew, ynew, color='blue')
            #     ax_stripes_arr[0].scatter(np.linspace(0, stripe.shape[0]-1, stripe.shape[0]), stripe)
            #     ax_stripes_arr[0].plot(np.linspace(0, stripe.shape[0] - 1, stripe.shape[0]), np.abs(img[x_stripe, :]), color='yellow')
            #     ax_stripes_arr[1].plot(np.arange(0, stripe.shape[0]), raw_img[x_stripe,:], color='black')
            #     ax_stripes_arr[2].plot(np.arange(0, stripe.shape[0]-1)+0.5, np.abs(np.diff(raw_img[x_stripe, :])), color='grey')
            #     # ax_stripes_arr[2].plot(zone_borders[0]+0.5+np.arange(0, stripe2.shape[0]), fit_result.best_fit, color='blue')
            #     if not time_deriv_only:
            #         ax_stripes_arr[2].plot(zone_borders[0] + 0.5 + np.arange(0, stripe2.shape[0]), stripe2,
            #                                color='yellow')
            #         ax_stripes_arr[2].axvline(x=main_peak, color='green')
            #     for p in peaks:
            #         ax_stripes_arr[0].axvline(x=p/upsampling_factor, color='red')
            #     ax_stripes_arr[0].axvline(x= main_peak_in_time_derivative_stripe, color='green')
            #     # ax_stripes.plot(ynew, color='blue')
            #     # ax_stripes.axvline(x=main_peak*upsampling_factor, color='green')
            #     ax_stripes_arr[0].set_xlim( main_peak_in_time_derivative_stripe-40,  main_peak_in_time_derivative_stripe+40)
            #     # plt.show()
            #     fig_stripes.savefig(frames_directory + '\\delamination_fronts\\stripes\\{0}'.format(filename_for_stripe))
            return main_peak

        cross_x = []
        cross_y = []

        # f3 = plt.figure(3)
        # plt.imshow(img)
        for i in range(np.shape(img)[0]):
            # line = np.abs(img[:, i])[::-1]
            # if k < istart + 20 and i == 250 and testing:
            #     plt.plot(line)
            #     plt.show()
            #     print('Yoohoo!')
            # intercept = (line[:] > c).argmax()
            # if intercept > 0 and i < 250 and i > 10:
            #     cross_x.append(i)
            #     cross_y.append(np.shape(img)[0] - intercept)
        # s = UnivariateSpline(cross_x, cross_y, k=1, s=1)
            front_position = get_front_location_in_stript(i, filename_for_stripe='{0:08d}.png'.format(k))
            if front_position >= 0:
                cross_x.append(i)
                cross_y.append(front_position)
                # plt.scatter(front_position, i, s=1, color='red')
        # plt.show()
        #     ax_fronts.plot(cross_y, cross_x, color='blue', alpha=0.2)
        front_curve = np.stack([np.array(cross_x), np.array(cross_y)])
        front_curves.append(front_curve)

    pickle.dump(front_curves, open(frames_directory + '\\delamination_fronts\\fronts\\front_curves_spacederiv.p', "wb"))
    pickle.dump(roi, open(frames_directory + '\\delamination_fronts\\fronts\\roi.p', "wb"))
    pickle.dump([istart, iend], open(frames_directory + '\\delamination_fronts\\fronts\\istart_iend.p', "wb"))
    print('Execution time: {0} min'.format((time.time() - t0) / 60))


# find_fronts_on_frames()

pickle.dump([istart, iend], open(frames_directory + '\\delamination_fronts\\fronts\\istart_iend.p', "wb"))



# plt.show()

    # ALL KINDS OF FILTERING
    # if len(cross_x) < 5:
    #     if not testing:
    #         print('Pass on frame {0}'.format(k))
    #     continue
    # # remove outliers
    # dy = np.diff(cross_y)
    # bads = np.where(np.abs(dy) > 10)[0]
    # bads = np.concatenate((bads, (bads + 1)), axis=0)
    # cross_x = np.delete(cross_x, bads)
    # cross_y = np.delete(cross_y, bads)
    #
    # bads = np.where(cross_y > 270)[0]
    # #    bads = np.concatenate((bads, (bads+1)), axis=0)
    # cross_x = np.delete(cross_x, bads)
    # cross_y = np.delete(cross_y, bads)
    #
    # tck = splrep(cross_x, cross_y, s=len(cross_x))
    # #    plt.plot(cross_x, cross_y, linewidth=0.8)
    # xs = np.linspace(min(cross_x), max(cross_x), 200)
    # ys = splev(xs, tck)
    # if not testing:
    #     if k % 10 == 0:
    #         print('Frame {0} is done.'.format(k))
    #         plt.plot(xs, ys, color='blue', linewidth=0.25, alpha=0.5)
#
# if not testing:
#     plt.xlim([0, 300])
#     plt.ylim([0, 300])
#     plt.savefig('front_family.eps', dpi=800)
#     plt.savefig('front_family.png', dpi=800)

import numpy as np
import skimage as sk
import matplotlib as mpl
mpl.rcParams['errorbar.capsize'] = 3
import matplotlib.pyplot as plt
import pylab
import psds
from scipy import optimize

def plot_power_spectrum(image, do_draw=True):
    pix_per_mm = 1
    im = image
    hanning = True
    freqG, psdG = psds.power_spectrum(im, oned=True, hanning=hanning)
    data_for_export = np.vstack((freqG * pix_per_mm, psdG))
    if do_draw:
        pylab.figure(2)
        pylab.clf()
        pylab.loglog(freqG * pix_per_mm, psdG, label='Power spectrum')

    def test_func(x, a):
        return a * (x)**(-2)

    params, params_covariance = optimize.curve_fit(test_func,
                                                   freqG * pix_per_mm, psdG,
                                                   p0=[2])
    if do_draw:
        pylab.loglog(freqG * pix_per_mm, test_func(freqG * pix_per_mm, params[0]), label='y~1/x$^2$')

    def test_func(x, a, b):
        return a + b*x
    cut_from_end = int(round(0.6*freqG.shape[0]))
    cut_from_start = 3
    params, params_covariance = optimize.curve_fit(test_func,
                                                   np.log(freqG[cut_from_start:cut_from_end] * pix_per_mm),
                                                   np.log(psdG[cut_from_start:cut_from_end]),
                                                   sigma=np.sqrt(freqG[cut_from_start:cut_from_end]),
                                                   p0=[2, -2])
    if do_draw:
        pylab.loglog(freqG * pix_per_mm,
                     np.exp(test_func(np.log(freqG * pix_per_mm), params[0], params[1])),
                     label='Fit power law, $\\alpha$={0:.2f}'.format(params[1]))

        pylab.legend(loc='best')
        pylab.xlabel("Spatial frequency ($px^{-1}$)")
        pylab.ylabel("Power spectral density, normalized")
        pylab.grid(True, which="both", ls="-")
        plt.show()
    return params[1]

def make_image(N, K, alpha=2):
    image = np.zeros(shape=(N, N), dtype=np.float)
    for i in range(K):
        # radius
        R = 1+np.random.random()*(N*0.2)
        #circle center coordinates
        # x0 = R + np.random.random()*(N-2*R)
        # y0 = R + np.random.random()*(N-2*R)
        x0 = np.random.random()*N
        y0 = np.random.random()*N
        # amplitude
        amplitude = 1/(R)**alpha
        # make circle and add to image
        rr, cc = sk.draw.circle(x0, y0, radius=R, shape=(N,N))
        image[rr, cc] += amplitude
    return image

def make_one_circle_in_the_center():
    image = np.zeros(shape=(1000, 1000), dtype=np.float)
    rr, cc = sk.draw.rectangle((450, 450), extent=(100, 100))
    image[rr, cc] += 1
    return image

if __name__ == '__main__':
    np.random.seed(1)
    f1 = plt.figure(1)
    image = make_image(1000, 200)

    # thresh = 0.005 * (image.max() - image.min())
    # image[image>thresh] = 1
    # image[image<=thresh] = 0
    # image[image<thresh] = thresh
    image = image**(0.5)

    plt.imshow(image, cmap='Greys', vmax=np.median(image)*5)
    # plt.imshow(image, cmap='Greys')
    plt.xlabel('pixels')
    plt.ylabel('pixels')
    f_spec1 = plt.figure(2)

    plot_power_spectrum(image)
    plt.show()

    target_folder ='theoretical_model/2d_discs/'
    # np.random.seed(1)
    # alphas = np.linspace(0.5, 3, 30)
    # gammas = []
    # gamma_stds = []
    # for n, alpha in enumerate(alphas):
    #     print(alpha)
    #     gammas_here = []
    #     number_of_repetitions = 5*(n+1)
    #     for k in range(number_of_repetitions):
    #         image = make_image(1000, 200, alpha=alpha)
    #         gamma = plot_power_spectrum(image, do_draw=False)
    #         gammas_here.append(gamma)
    #     gammas.append(np.mean(np.array(gammas_here)))
    #     gamma_stds.append(np.std(np.array(gammas_here)))
    # np.save(target_folder + 'alphas', np.array(alphas))
    # np.save(target_folder + 'gammas', np.array(gammas))
    # np.save(target_folder + 'gamma_stds', np.array(gamma_stds))

    alphas = np.load(target_folder + 'alphas.npy')
    gammas = np.load(target_folder + 'gammas.npy')
    gamma_stds = np.load(target_folder + 'gamma_stds.npy')

    f1 = plt.figure(1, figsize=(10, 4))

    plt.errorbar(x=alphas, y=gammas, yerr=gamma_stds, marker='o')
    # plt.plot(alphas, gammas)
    plt.xlabel('Exponent $\\alpha$ in power law $\\Delta\\sigma \propto$R$^{-\\alpha}$ of charge density magnitude $\\Delta\\sigma$ vs. radius R')
    plt.ylabel('Best-fit power spectral density exponent $\\beta$')
    plt.axhline(y=-2, color='black', linestyle='--')
    plt.tight_layout()
    f1.savefig('theoretical_model/figures/exponent_vs_exponent.png', dpi=300)
    plt.show()
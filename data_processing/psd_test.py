import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.gridspec as gridspec
# from scipy.stats import norm
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# dt = 0.01
# t = np.arange(0, 10, dt)
# nse = np.random.randn(len(t))
# r = np.exp(-t / 0.05)
#
# cnse = np.convolve(nse, r) * dt
# cnse = cnse[:len(t)]
# s = 0.1 * np.sin(2 * np.pi * t) + cnse
#
# plt.subplot(211)
# plt.plot(t, s)
# plt.subplot(212)
# plt.psd(s, 512, 1 / dt)
#
# plt.show()

# from scipy.fftpack import fft, fftfreq
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)
# plt.plot(xf)
# plt.grid()
# plt.show()

for alpha in np.linspace(1, 3, 100):
    x = np.linspace(100, 200, 10)
    plt.plot(x, x**alpha)
    plt.show()
# f1, axarr = plt.subplots(3)
#
# xs = np.linspace(-10, 10, 1000)
# ys = norm.pdf(xs,0,1)
# axarr[0].plot(xs, ys, label='Женщины')
# ys = norm.pdf(xs,0,1.1)
# axarr[0].plot(xs, ys, label='Мужчины')
# axarr[0].legend()
#
# ys = norm.pdf(xs,0,1)
# axarr[1].plot(xs, ys, label='Женщины')
# ys = norm.pdf(xs,0,1.1)
# axarr[1].plot(xs, ys, label='Мужчины')
# axarr[1].set_yscale('log')
# axarr[1].set_ylabel('Лог. масштаб')
#
#
# ys = norm.pdf(xs,0,1.1)/(norm.pdf(xs,0,1.1) + norm.pdf(xs,0,1))
# axarr[2].plot(xs, ys*100, color='black', label='Доля мужчин')
# axarr[2].set_ylabel('%')
# # axarr[1].set_yscale('log')
#
# plt.legend()
# plt.show()


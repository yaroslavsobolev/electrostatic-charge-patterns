import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev, splrep
from scipy.interpolate import PPoly
import pickle

f1 = plt.figure(1, figsize=(5,5))

data = np.loadtxt('hysteresis/horn_smith_1992_science.txt', skiprows=1, delimiter='\t')
xs = []
ys = []
for i in range(4):
    gap = 1/2*(data[i, 0] + data[i, 2])
    hyst = data[i, 3]/data[i, 1]
    xs.append(gap)
    ys.append(hyst)
plt.scatter(xs, ys, label='Horn & Smith, 1992')

data = np.loadtxt('hysteresis/horn_smith_1993_nature.txt', skiprows=1, delimiter='\t')
xs = []
ys = []
for i in range(4):
    gap = 1/2*(data[i, 0] + data[i, 2])
    hyst = data[i, 3]/data[i, 1]
    xs.append(gap)
    ys.append(hyst)

plt.scatter(xs, ys, label='Horn, Smith, & Grabbe, 1993')


data = np.loadtxt('hysteresis/Kwetkus_Sattler_Siegmann__1992_epoxy-brass_260mbar_zoom.txt', skiprows=1, delimiter=',')
xs = []
ys = []
for i in range(data.shape[0]):
    gap = 15*260/1013.25 #um*atm
    hyst = data[i, 3]/data[i, 1]
    xs.append(gap)
    ys.append(hyst)

plt.scatter(xs, ys, marker='x', label='Kwetkus, Sattler & Siegmann, 1992', alpha=0.6, color='black')

plt.xlabel('Gap-pressure product, $\mu$m$\cdot$atm')
plt.ylabel('Ratio of critical electric fields:\nextinction limit to breakdown limit')
plt.ylim([0, 1])
plt.legend()

f2 = plt.figure(2, figsize=(2.3,4.45))

data = np.loadtxt('hysteresis/McCarthy_Whitesides_2006__ionic_electrets.txt', skiprows=1, delimiter='\t')
xs = []
ys = []
for i in range(4):
    hyst = data[i, 3]/data[i, 1]
    xs.append(0)
    ys.append(hyst)

plt.scatter(xs, ys, color='C2', label='McCarthy &\nWhitesides,\n2007')
plt.tight_layout()
plt.legend()

plt.ylim([0, 1])

f1.savefig('figures/breakdown_hysteresis_1.png', dpi=400)
f2.savefig('figures/breakdown_hysteresis_2.png', dpi=400)
plt.show()
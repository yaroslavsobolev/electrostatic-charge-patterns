import numpy as np
import matplotlib.pyplot as plt

# cols_nums = list(range(2,31))
xs = np.loadtxt('XPS_data/191213__colx.txt')
sil_raw = np.loadtxt('XPS_data/191213__Si2p.txt', skiprows=3)
oxy_raw = np.loadtxt('XPS_data/191213__O1s.txt', skiprows=3)
ys = sil_raw[:,0]
sil = sil_raw[:,1:]
oxy = oxy_raw[:,1:]
oxy[oxy<0] = np.nan
sil[sil<0] = np.nan
sil_norm = sil/oxy
plt.imshow(sil_norm, vmax=0.5, vmin=0)
plt.show()
print(1)

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

target_folder = 'XPS_data/XPS_everything_data/spectra/'

# samples = ['A2NotCharged',
#            'BigPimple',
#            'BigPimple2',
#            'boundary',
#            'ChargedNoPatt',
#            'Discharge1',
#            'Discharge2',
#            # 'Discharge3',
#            'discharge4',
#            'MultiPimple2',
#            'MultiPimple3',
#            'MultiPimples',
#            'NotCharged',
#            # 'Pimple1',
#            'TipOfTheShip'
#            ]

samples = [
    'ChargedNoPatt',
    'BigPimple2',
           'BigPimple',
           'Discharge2',
'MultiPimples',
'discharge4',
'MultiPimple2',
           'Discharge1',
'MultiPimple3',
'TipOfTheShip',
    'boundary',
'A2NotCharged',
'NotCharged'
           ]

for i in reversed(list(range(0, len(samples), 2))):
    print(samples[i])
    print('\n\n\n')

elements = ['Si', 'O', 'C']
# normalization = [60000, 60000, 0.3, 4000]
normalization = [51019.63360000001, 47775.86, 0.2856979495129698, 3120.0685999999987]
# normalization = [1]*4
alphas = [0.4, 0.4, 1, 1]
# energies = [, ]

def open_file_and_get_one_value(filename, x_location):
    data = np.loadtxt(filename, skiprows=3, delimiter='\t')
    f = interpolate.interp1d(data[:,0], data[:,1])
    # res = np.interp(x_location, data[:,0], data[:,1])
    res = f(x_location)
    # print(res)
    # plt.plot(data[:,0], data[:,1])
    # xs = np.linspace(96, 102, 100)
    # plt.plot(xs, f(xs))
    # plt.show()
    return res

def open_file_and_get_max(filename):
    data = np.loadtxt(filename, skiprows=3, delimiter='\t')
    return np.max(data[:,1])

fig, ax = plt.subplots(figsize=(2.5,9))
N = len(samples)
# fig, axarr = plt.subplots(1, N)
labels = ['C1S', 'O1s', 'O1s Asymmetry', 'Si2p']
# for i, label in enumerate(labels):

# Carbon
i = 0
label = labels[i]
signals = []
for sample in samples:
    signal = open_file_and_get_max(target_folder + sample + '_C.txt')
    bkg = open_file_and_get_one_value(target_folder + sample + '_C.txt',
                                      x_location=289)
    signals.append((signal-bkg)/normalization[i])
print(np.max(np.array(signals)))
plt.barh(6*np.arange(N)+i, signals, label=label, alpha=alphas[i])

# Oxygen
i = 1
label = labels[i]
signals = []
diffs = []
for sample in samples:
    left_peak = open_file_and_get_one_value(target_folder + sample + '_O.txt',
                                      x_location=531.98)
    right_peak = open_file_and_get_one_value(target_folder + sample + '_O.txt',
                                      x_location=530.48)
    bkg = open_file_and_get_one_value(target_folder + sample + '_O.txt',
                                      x_location=534.78)
    signal = 0.5*( (left_peak-bkg) +  (right_peak-bkg) )
    diff = (right_peak - left_peak) / signal
    signals.append(signal/normalization[i])
    diffs.append(diff/normalization[i+1])
plt.barh(6*np.arange(N)+i, signals, label=label, alpha=alphas[i])
plt.barh(6*np.arange(N)+i+1, diffs, label='O1s asymmetry', alpha=alphas[i+1])
print(np.max(np.array(signals)))
print(np.max(np.array(diffs)))

# Silicon
i = 3
label = labels[i]
signals = []
for sample in samples:
    signal = open_file_and_get_max(target_folder + sample + '_Si.txt')
    bkg = open_file_and_get_one_value(target_folder + sample + '_Si.txt',
                                      x_location=104)
    signals.append((signal-bkg)/normalization[i])
print(np.max(np.array(signals)))
plt.barh(6*np.arange(N)+i, signals, label=label, alpha=alphas[i])

# for i in range(N):
#     plt.axhline(y=i*6-1)
# plt.legend()
fig.savefig('all_XPS.png', dpi=300, transparent=True)
plt.show()

plt.scatter(diffs, signals, alpha=0.5)
z = np.polyfit(diffs, signals, 1)
xp = np.linspace(0, 1, 100)
plt.plot(xp, np.poly1d(z)(xp), color='grey', zorder=-10, alpha=0.7)
plt.xlabel('Presence of air discharge plasma\n(asymmetry of the O1s peak, a.u.)')
plt.ylabel('Material transfer\n(magnitude of Si2p peak, a.u.)')



plt.show()
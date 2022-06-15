import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import integrate

gravity=9.79776 #m/s^2 -- this is local gravity in Ulsan, Republic of Korea

def generate_lines(filePath, delimiters=[]):
    with open(filePath) as f:
        for line in f:
            line = line.strip()  # removes newline character from end
            for d in delimiters:
                line = line.replace(d, " ")
            yield line

def get_adhesion_work(target_folder, do_plotting=True, plotname='default.png'):
    try:
        file_name = glob.glob(target_folder + "force/force_vs_time.txt")[0]
    except IndexError:
        print('FORCE FILE NOT FOUND IN {0}'.format(target_folder))
        return np.nan
    gen = generate_lines(file_name, delimiters=['\t', ' '])
    data = np.loadtxt(gen, usecols=[0,1,2], skiprows=2) # time (s), position (mm), force (grams)
    # Find where delamination begins -- we assume that it is the point of maximum force
    delamination_start_index = np.argmax(data[:,2])
    # find where delamination ends. We assume that it is the first positive force from the end of trace
    delamination_end_index = data.shape[0] - 1 - np.argmax(np.flip(data[:,2])>0)
    delamination_time = data[delamination_end_index, 0] - data[delamination_start_index, 0]
    # integrate with simps
    force = gravity*(0.001)*data[delamination_start_index:delamination_end_index + 1, 2] # in Newtons
    coordinate = -0.001*data[delamination_start_index:delamination_end_index + 1, 1] # in meters
    work = integrate.trapz(y=force, x=coordinate)
    area = 5e-4 # 5 cm^2 in m^2
    work_per_area = work/area
    print('Work per area: {0:.2f} mJ/m^2, time={1:.2f}'.format(work_per_area/1e-3, delamination_time))
    fig, ax = plt.subplots()
    plt.plot(-1*data[:,1], data[:,2], color='black')
    plt.plot(-1*data[delamination_start_index:delamination_end_index + 1,1],
             data[delamination_start_index:delamination_end_index + 1,2], color='yellow')
    plt.axhline(y=0, color='grey')
    plt.axvline(x=-1*data[delamination_start_index, 1], color='red')
    plt.axvline(x=-1*data[delamination_end_index, 1], color='blue')
    plt.ylabel('Force, grams')
    plt.xlabel('Position, mm')
    fig.savefig('force_curve_figures/per_experiment/{0}.png'.format(plotname))
    # plt.show()
    plt.close(fig)
    return work_per_area, delamination_time


if __name__ == '__main__':
    # work = get_adhesion_work('Y:/PDMS-PMMA_delamination_experiments/data/' \
    #                 '20191017_5cm_3in_62RH_eq30min_oldUVPDMS_PMMAtol_uniformspeeds_0p1_B01/')

    directories = glob.glob("Y:/PDMS-PMMA_delamination_experiments/data/*3in*newPDMS*/")
    directories = [x.replace('\\', '/') for x in directories if ('Argo' not in x) and ('forAFM' not in x)]

    works = []
    for n, target_dir in enumerate(directories):
        # if n < 84:
        #     continue
        print('{0} >>> {1}'.format(n, target_dir))
        # try:
        work_per_area, delamination_time = get_adhesion_work(target_dir, plotname='{0:03d}'.format(n))
        works.append(work_per_area)

    np.savetxt('force_curve_figures/works_per_area.txt', np.array(works))
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import os
import scipy.signal as signal



cwd = os.getcwd()
print(cwd)

with open("../../OriginalData/CarouselExperimentsData/day1_all_data_cleaned_updated.obj", "rb") as input_file:
    PFC_DA_quinine_data = pickle.load(input_file)

with open("../../OriginalData/CarouselExperimentsData/day1_all_data_cleaned_updated.obj", "rb") as input_file:
    NAc_DA_quinine_data = pickle.load(input_file)

for key in NAc_DA_quinine_data.keys():
    print(key)



data_keys = ['water', 'Q1', 'Q2', 'Q3', 'Q4']

ts = NAc_DA_quinine_data['time']


sos = signal.butter(4, 2, btype='low', fs=1/(ts[1] - ts[0]), output='sos')

data_list = [NAc_DA_quinine_data[key] for key in data_keys]

# crop the windowed data so it has consistent lengths and convert to a numpy array
min_index = data_list[0][0][0].shape[0]
for i in range(len(data_list)):
    for j in range(len(data_list[i])):
        for k in range(len(data_list[i][j])):
            min_index = np.minimum(min_index, data_list[i][j][k].shape[0])

ts = ts[:min_index]

for i in range(len(data_list)):
    for j in range(len(data_list[i])):
        for k in range(len(data_list[i][j])):
            data_list[i][j][k] = data_list[i][j][k][:min_index]
        data_list[i][j] = np.array(data_list[i][j])


# lowpass filter and baseline the data

for i in range(len(data_list)):
    for j in range(len(data_list[i])):
        for k in range(len(data_list[i][j])):
            data_list[i][j][k, :] = signal.sosfiltfilt(sos, data_list[i][j][k, :])
            data_list[i][j][k, :] = data_list[i][j][k, :] - np.mean(data_list[i][j][k, ts < -2])

print('length of data_list =', len(data_list))

figure, axes = plt.subplots(10, 5)
for animal_number in range(10):

    mx = []
    mn = []
    for condition in range(5):
        number_of_trials = data_list[condition][animal_number].shape[0]
        mx.extend([np.max(data_list[condition][animal_number][trial_number, :]) for trial_number in range(number_of_trials)])
        mn.extend([np.min(data_list[condition][animal_number][trial_number, :]) for trial_number in range(number_of_trials)])

    mx = np.max(mx)
    mn = np.min(mn)

    mid = (mx + mn)/2
    dy = mx - mid

    for condition in range(5):
        number_of_trials = data_list[condition][animal_number].shape[0]

        for trial_number in range(len(data_list[condition][animal_number])):
            axes[animal_number, condition].plot(ts, data_list[condition][animal_number][trial_number, :], color='k', linewidth=0.5)
            axes[animal_number, condition].set_ylim([mid - dy*1.1, mid + dy*1.1])
            prettyplot.no_box(axes[animal_number, condition])



plt.show()
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import os
import scipy.signal as signal
from scipy.stats import sem
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests as mult_test


data_directory_path_day_1 = '../../OriginalData/FearConditioningData/Day1/mPFC/'
data_directory_path_day_2 = '../../OriginalData/FearConditioningData/Day2/mPFC/'

shock_duration = 0.5


def get_data(data_directory_path):

    file_names = os.listdir(data_directory_path)

    data = []
    for file_name in file_names:
        with open(data_directory_path + file_name, "rb") as input_file:
            data.append(pickle.load(input_file))
            print('loaded file,', file_name)

    return data

data_list = get_data(data_directory_path_day_1)


for key in data_list[0].keys():
    print(key)



def fill_event_ranges(event_times, event_duration, color, axes=None):
    for i in range(len(event_times)):
        if axes is None:
            plt.axvspan(event_times[i], event_times[i] + event_duration, color=color)
        else:
            axes[i].axvspan(event_times[i], event_times[i] + event_duration, color=color)

def plot_example_dataset(data):

    fig = plt.figure(figsize=(12, 2))
    plt.subplots_adjust(bottom=0.2)
    t_0 = data['tone1'][0]

    fill_event_ranges((data['tone1'] - t_0)/60.0, 5/60.0, color=[1, 0.5, 0.5])
    fill_event_ranges((data['tone_test'] - t_0)/60.0, 5/60.0, color=[1, 0.5, 0.5])

    fill_event_ranges((data['shock'] - t_0)/60.0, shock_duration/60.0, color=prettyplot.colors['red'])

    fill_event_ranges((data['tone2'] - t_0)/60.0, 5/60.0, color=[0.7, 0.7, 0.7])
    ts = data['ts'] - data['tone1'][0]
    ts = ts/60.0
    plt.plot(ts, 100*data['F_ex'], color='k')
    prettyplot.xlabel('min')
    prettyplot.ylabel('%dF/F')

    plt.xlim([-1, 35])
    plt.ylim([-25, 60])
    prettyplot.no_box()


plot_example_dataset(data_list[0])
plt.savefig('ToneShockExampleRecording.pdf', transparent=True)
plt.show()


def collect_trial_windows(data_list, event, window):

    window_list = [] * len(data_list)

    fs = data_list[0]['fs']
    index_window = [int(np.round(window[0] * fs)), int(np.round(window[1] * fs))]

    for data in data_list:
        within_data_windows = []
        events = data[event]

        for i in range(len(events)):

            t_event = events[i]

            index_event = np.argmin(np.square(data['ts'] - t_event))
            within_data_windows.append(data['F_ex'][index_event + index_window[0]:index_event + index_window[1]])
            ts = data['ts'][index_event + index_window[0]:index_event + index_window[1]] - data['ts'][index_event]

        window_list.append(within_data_windows)

    window_list = np.array(window_list)
    sos = signal.bessel(4, 2, 'low', fs=fs, output='sos')
    window_list = signal.sosfiltfilt(sos, window_list, axis=2)

    return ts, window_list





def plot_example_window_list(ts, window_list, figsize, fillregion, fillcolor):

    fig, axes = plt.subplots(1, len(window_list), figsize=figsize)
    plt.subplots_adjust(left=0.1, bottom=0.2)


    for i in range(window_list.shape[0]):
        if type(fillregion[0]) != list:
            fillregion = [fillregion]
            fillcolor = [fillcolor]

        for k, region in enumerate(fillregion):
            axes[i].axvspan(region[0], region[1], color=fillcolor[k])

        axes[i].plot(ts, window_list[i, :], color='k')
        if i == 0:
            prettyplot.no_box(axes[i])
        else:
            prettyplot.x_axis_only(axes[i])


def plot_window_list_average(ts, window_list, figsize, fillregion, fillcolor, yrange):

    fig, axes = plt.subplots(1, window_list.shape[1], figsize=figsize)
    plt.subplots_adjust(left=0.1, bottom=0.2)


    for i in range(window_list.shape[1]):
        if type(fillregion[0]) != list:
            fillregion = [fillregion]
            fillcolor = [fillcolor]

        for k, region in enumerate(fillregion):
            axes[i].axvspan(region[0], region[1], color=fillcolor[k])
        print(window_list.shape)
        baselined = window_list[:, i, :] - np.mean(window_list[:, i, ts < 0], 1, keepdims=True)
        sem = np.std(baselined, 0)/np.sqrt(baselined.shape[0])
        mean = np.mean(baselined, 0)
        axes[i].fill_between(ts, mean[:] - sem[:], mean[:] + sem[:], color=[0.8, 0.8,   0.8])
        axes[i].plot(ts, mean[:], color='k')
        axes[i].set_ylim(yrange)
        if i == 0:
            prettyplot.no_box(axes[i])
        else:
            prettyplot.x_axis_only(axes[i])


def plot_window_list_average_with_maximum(ts, window_list, figsize, fillregion, fillcolor, yrange):

    fig, axes = plt.subplots(1, window_list.shape[1], figsize=figsize)
    plt.subplots_adjust(left=0.1, bottom=0.2)

    for i in range(window_list.shape[1]):
        if type(fillregion[0]) != list:
            fillregion = [fillregion]
            fillcolor = [fillcolor]

        for k, region in enumerate(fillregion):
            axes[i].axvspan(region[0], region[1], color=fillcolor[k])
        print(window_list.shape)
        baselined = window_list[:, i, :]



        sem = np.std(baselined, 0)/np.sqrt(baselined.shape[0])
        mean = np.mean(baselined, 0)


        mean_filtered = signal.sosfiltfilt(sos, mean)

        axes[i].fill_between(ts, mean[:] - sem[:], mean[:] + sem[:], color=[0.8, 0.8,   0.8])
        axes[i].plot(ts, mean[:], color='k')
        axes[i].plot(ts, mean_filtered)
        axes[i].set_ylim(yrange)

        axes[i].axhline(np.mean(maxima[:, i]), color='k')

        if i == 0:
            prettyplot.no_box(axes[i])
        else:
            prettyplot.x_axis_only(axes[i])
    return maxima

ts, tone1_window_list = collect_trial_windows(data_list, 'tone1', [-5, 15])
tone1_window_list = tone1_window_list - np.mean(tone1_window_list[:, :, ts < 0], axis=2, keepdims=True)
maxima_tone1 = np.max(tone1_window_list[:, :, ts < 4.5], axis=2)

ts, tone_test_window_list = collect_trial_windows(data_list, 'tone_test', [-5, 15])
tone_test_window_list = tone_test_window_list - np.mean(tone_test_window_list[:, :, ts < 0], axis=2, keepdims=True)
maxima_tone_test = np.max(tone_test_window_list, axis=2)


# plot_example_window_list(ts, tone1_window_list[0], figsize=(10, 2), fillregion=[[0, 5], [5, 6]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]])
# plt.savefig('ToneShockPeriEventSingleAnimalExample.pdf', transparent=True)
# plt.show()

plot_window_list_average(ts, tone1_window_list, figsize=(10, 2), fillregion=[[0, cue_duration], [cue_duration, cue_duration + shock_duration]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]], yrange=[-.05, .3])
plt.savefig('ToneShockPeriEvent.pdf', transparent=True)
plt.show()

ts, tone2_window_list = collect_trial_windows(data_list, 'tone2', [-5, 15])
tone2_window_list = tone2_window_list - np.mean(tone2_window_list[:, :, ts < 0], axis=2, keepdims=True)
maxima_tone2 = np.max(tone2_window_list[:, :, ts < 4.5], axis=2)

plot_window_list_average(ts, tone2_window_list, figsize=(10, 2), fillregion=[[0, cue_duration], [cue_duration, cue_duration + shock_duration]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]], yrange=[-.05, .3])
plt.savefig('ToneShockPeriEventDistractorTone.pdf', transparent=True)
plt.show()
#
#
plot_window_list_average(ts, tone_test_window_list[:, [0, 4, 9, 14, 19, 24, 29], :], figsize=(10, 2), fillregion=[[0, cue_duration], [cue_duration, cue_duration + shock_duration]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]], yrange=[-.05, .3])
plt.savefig('ToneShockExtinctionTrials151015etc.pdf', transparent=True)
plt.show()

data_list_day_2 = get_data(data_directory_path_day_2)
ts_day_2, tone_test_window_list_day_2 = collect_trial_windows(data_list_day_2, 'tone_test', [-5, 15])
tone_test_window_list_day_2 = tone_test_window_list_day_2 - np.mean(tone_test_window_list_day_2[:, :, ts < 0], axis=2, keepdims=True)
maxima_tone_test_day_2 = np.max(tone_test_window_list_day_2, axis=2)
# plot_window_list_average(ts_day_2, tone_test_window_list_day_2[:, [0, 4, 9, 14, 19, 24, 29], :], figsize=(10, 2), fillregion=[[0, cue_duration], [cue_duration, cue_duration + shock_duration]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]], yrange=[-.05, .3])
# plt.savefig('ToneShockExtinctionTrials151015etcDay2.pdf', transparent=True)

plt.show()

fig = plt.figure(figsize=(8, 2))
plt.subplots_adjust(bottom=0.2)

offset = 1
# for i in range(maxima_tone1.shape[0]):
#     plt.plot(np.arange(maxima_tone1.shape[1]) + offset, maxima_tone1[i, :], color=[0.8, 0.8, 0.8], linestyle='', marker='.')
plt.errorbar(np.arange(maxima_tone2.shape[1]) + offset, np.mean(maxima_tone2, 0), yerr=sem(maxima_tone2, 0, ddof=0), color=[0.5, 0.5, 0.5], linestyle='', marker='o', capsize=4)

plt.errorbar(np.arange(maxima_tone1.shape[1]) + offset, np.mean(maxima_tone1, 0), yerr=sem(maxima_tone1, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4)

offset += maxima_tone1.shape[1] + 5

# for i in range(maxima_tone_test.shape[0]):
#     plt.plot(np.arange(maxima_tone_test.shape[1]) + offset, maxima_tone_test[i, :], color=[0.8, 0.8, 0.8], linestyle='', marker='.')
plt.errorbar(np.arange(maxima_tone_test.shape[1]) + offset, np.mean(maxima_tone_test, 0), yerr=sem(maxima_tone_test, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4)
offset += maxima_tone_test.shape[1] + 5

# for i in range(maxima_tone_test_day_2.shape[0]):
#     plt.plot(np.arange(maxima_tone_test_day_2.shape[1]) + offset, maxima_tone_test_day_2[i, :], color=[0.8, 0.8, 0.8], linestyle='', marker='.')
plt.errorbar(np.arange(maxima_tone_test_day_2.shape[1]) + offset, np.mean(maxima_tone_test_day_2, 0), yerr=sem(maxima_tone_test_day_2, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4)

plt.xticks([1, 8,
            13 + 1, 13 + 5, 13 + 10, 13 + 15, 13 + 20, 13 + 25, 13 + 30,
            48 + 1, 48 + 5, 48 + 10, 48 + 15, 48 + 20, 48 + 25, 48 + 30,
            ],
           ['1', '8', '1', '5', '10', '15', '20', '25', '30', '1', '5', '10', '15', '20', '25', '30'])
prettyplot.no_box()
plt.ylim([0, .16])
plt.savefig('ToneShockToneEvokedResponseAcrossTrials.pdf', transparent=True)
plt.show()

z = maxima_tone1

maxima_tone1 = maxima_tone1[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]  # There's one mouse that didn't get run on the last day. Exclude it for pairwise t-tests
maxima_tone_test = maxima_tone_test[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]

print(maxima_tone1.shape, maxima_tone_test.shape, maxima_tone_test_day_2.shape)
firsts_and_lasts = np.array([maxima_tone1[:, 0], maxima_tone1[:, -1], maxima_tone_test[:, 0], maxima_tone_test[:, -1], maxima_tone_test_day_2[:, 0], maxima_tone_test_day_2[:, -1]])
t_tests = []
for i in range(firsts_and_lasts.shape[0] - 1):
    stats = ttest_rel(firsts_and_lasts[i, :], firsts_and_lasts[i + 1, :])
    t_tests.append(stats.pvalue)

stats = ttest_rel(maxima_tone2[:, 0], maxima_tone2[:, -1])
print('tone 2 p-value =', stats.pvalue)

stats = ttest_rel(z[:, -1], maxima_tone2[:, -1])
print('tone 1 vs. tone 2 p-value =', stats.pvalue)

print(t_tests)

corrected_pvalues = mult_test(t_tests, method='bonferroni')[1]

print(corrected_pvalues)


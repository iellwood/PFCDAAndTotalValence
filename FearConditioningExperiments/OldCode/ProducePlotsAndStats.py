import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import utils.statsfunctions as statsfunctions
import os
import scipy.signal as signal
from scipy.stats import sem
from scipy.stats import ttest_rel
from scipy.stats import linregress

from statsmodels.stats.multitest import multipletests as mult_test



def perform_analysis(data_directory_path_day_1, data_directory_path_day_2, brain_region, CS_name_day_2):

    header = 'FigurePdfs/MainPlots/FearConditioning_' + brain_region + '_'

    cue_duration = 5
    shock_duration = 0.5
    window = [-20, 20]
    averaging_window = [0.0, 4.5]

    def get_data(data_directory_path):

        file_names = os.listdir(data_directory_path)

        data = []
        for file_name in file_names:
            with open(data_directory_path + file_name, "rb") as input_file:
                data.append(pickle.load(input_file))
        return data

    data_list = get_data(data_directory_path_day_1)


    def fill_event_ranges(event_times, event_duration, color, axes=None):
        for i in range(len(event_times)):
            if axes is None:
                plt.axvspan(event_times[i], event_times[i] + event_duration, color=color)
            else:
                axes[i].axvspan(event_times[i], event_times[i] + event_duration, color=color)

    def plot_example_dataset(data, time_shown=16):

        fig = plt.figure(figsize=(6, 2))
        plt.subplots_adjust(bottom=0.2)
        t_0 = data['tone1'][0]

        fill_event_ranges((data['tone1'] - t_0)/60.0, 5/60.0, color=[1, 0.5, 0.5])
        fill_event_ranges((data['tone_test'] - t_0)/60.0, 5/60.0, color=[1, 0.5, 0.5])

        fill_event_ranges((data['shock'] - t_0)/60.0, shock_duration/60.0, color=prettyplot.colors['red'])

        fill_event_ranges((data['tone2'] - t_0)/60.0, 5/60.0, color=[0.7, 0.7, 0.7])
        ts = data['ts'] - data['tone1'][0]
        ts = ts/60.0
        I = np.logical_and(ts > -1, ts < time_shown)
        plt.plot(ts[I], 100*data['F_ex'][I], color='k')
        prettyplot.xlabel('min')
        prettyplot.ylabel('%dF/F')

        plt.xlim([-1, time_shown])
        #plt.ylim([-25, 60])
        prettyplot.no_box()


    plot_example_dataset(data_list[0])
    plt.savefig(header + 'ConditioningExampleRecording.pdf', transparent=True)
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


    def plot_window_list_average(ts, window_list, figsize, fillregion, fillcolor, xrange=None, yrange=None, included_trials=None):

        if included_trials is None:
            fig, axes = plt.subplots(1, window_list.shape[1], figsize=figsize)
        else:
            fig, axes = plt.subplots(1, len(included_trials), figsize=figsize)
        plt.subplots_adjust(left=0.1, bottom=0.2)


        for i in range(len(axes)):
            if type(fillregion[0]) != list:
                fillregion = [fillregion]
                fillcolor = [fillcolor]
            if included_trials is None:
                trial_number = i
            else:
                trial_number = included_trials[i]
            for k, region in enumerate(fillregion):
                axes[i].axvspan(region[0], region[1], color=fillcolor[k])
            baselined = window_list[:, trial_number, :] - np.mean(window_list[:, trial_number, ts < 0], 1, keepdims=True)
            sem = np.std(baselined, 0)/np.sqrt(baselined.shape[0])
            mean = np.mean(baselined, 0)

            # Convert to percentages
            sem = 100*sem
            mean = 100*mean

            axes[i].fill_between(ts, mean[:] - sem[:], mean[:] + sem[:], color=[0.8, 0.8,   0.8])
            axes[i].plot(ts, mean[:], color='k')
            if yrange is not None:
                axes[i].set_ylim(yrange)
            if xrange is not None:
                axes[i].set_xlim(xrange)

            if i == 0:
                prettyplot.no_box(axes[i])
            else:
                prettyplot.x_axis_only(axes[i])



    # Compute the average response to the CS during conditioning

    ts, tone1_window_list = collect_trial_windows(data_list, 'tone1', window)
    # baseline the data
    tone1_window_list = tone1_window_list - np.mean(tone1_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_conditioning = np.mean(tone1_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    # Compute the average response to the CS during extinction

    ts, tone_test_window_list = collect_trial_windows(data_list, 'tone_test', window)
    # baseline the data
    tone_test_window_list = tone_test_window_list - np.mean(tone_test_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_extinction = np.mean(tone_test_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    # PLOT DAY 1 PERIEVENT DATA FOR CS/US DURING CONDITIONING

    plot_window_list_average(ts, tone1_window_list, included_trials=[0, 3, 7], figsize=(5, 2), fillregion=[[0, cue_duration], [cue_duration, cue_duration + shock_duration]], fillcolor=[[1, 0.85, 0.85], [1, 0.5, 0.5]], yrange=[-10, 30], xrange=[-5, 20])
    plt.savefig(header + 'Perievent_Conditioning_CS.pdf', transparent=True)
    plt.show()

    # PLOT DAY 1 PERIEVENT DATA FOR NCS DURING CONDITIONING

    ts, tone2_window_list = collect_trial_windows(data_list, 'tone2', window)
    # baseline the data
    tone2_window_list = tone2_window_list - np.mean(tone2_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_NCS_during_conditioning = np.mean(tone2_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    plot_window_list_average(ts, tone2_window_list, included_trials=[0, 3, 7], figsize=(5, 2), fillregion=[[0, cue_duration]], fillcolor=[[0.85, 0.85, 0.85]], yrange=[-10, 30], xrange=[-5, 20])
    plt.savefig(header + 'Perievent_Conditioning_NCS.pdf', transparent=True)
    plt.show()

    # PLOT DAY 1 PERIEVENT DATA FOR EXTINCTION

    plot_window_list_average(ts, tone_test_window_list, included_trials=[0, 14, 29], figsize=(5, 2), fillregion=[[0, cue_duration]], fillcolor=[[1, 0.85, 0.85]], yrange=[-10, 30], xrange=[-5, 20])
    plt.savefig(header + 'Perievent_Extinction_CS.pdf', transparent=True)
    plt.show()

    # PLOT DAY 2 PERIEVENT DATA

    data_list_day_2 = get_data(data_directory_path_day_2)
    ts_day_2, tone_test_window_list_day_2 = collect_trial_windows(data_list_day_2, CS_name_day_2, window)
    # Baseline the data
    tone_test_window_list_day_2 = tone_test_window_list_day_2 - np.mean(tone_test_window_list_day_2[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_extinction_memory_test = np.mean(tone_test_window_list_day_2[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)
    plot_window_list_average(ts_day_2, tone_test_window_list_day_2, included_trials=[0, 14, 29], figsize=(5, 2), fillregion=[[0, cue_duration]], fillcolor=[[1, 0.85, 0.85]], yrange=[-10, 30], xrange=[-5, 20])
    plt.savefig(header + 'Perievent_ExtinctionMemory_CS.pdf', transparent=True)

    plt.show()

    # Plot the summary

    fig = plt.figure(figsize=(8, 2))
    plt.subplots_adjust(bottom=0.2)

    offset = 1
    mean_response_to_NCS_during_conditioning *= 100
    plt.errorbar(np.arange(mean_response_to_NCS_during_conditioning.shape[1]) + offset, np.mean(mean_response_to_NCS_during_conditioning, 0), yerr=sem(mean_response_to_NCS_during_conditioning, 0, ddof=0), color=[0.5, 0.5, 0.5], linestyle='', marker='o', capsize=4)
    mean_response_to_CS_during_conditioning *= 100
    plt.errorbar(np.arange(mean_response_to_CS_during_conditioning.shape[1]) + offset, np.mean(mean_response_to_CS_during_conditioning, 0), yerr=sem(mean_response_to_CS_during_conditioning, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4)

    # Add the extinction data to the plot

    offset += mean_response_to_CS_during_conditioning.shape[1] + 5

    x = np.arange(mean_response_to_CS_during_extinction.shape[1]) + offset
    mean_response_to_CS_during_extinction *= 100
    plt.errorbar(x, np.mean(mean_response_to_CS_during_extinction, 0), yerr=sem(mean_response_to_CS_during_extinction, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4, zorder=0)
    offset += mean_response_to_CS_during_extinction.shape[1] + 5

    y = np.transpose(mean_response_to_CS_during_extinction, [1, 0])
    pvalue, Fvalue = statsfunctions.repeated_measures_ANOVA(x, y)
    Fvalue = statsfunctions.display_n_sig_figs(Fvalue, 2)
    pvalue = statsfunctions.display_n_sig_figs(pvalue, 1)
    plt.text(x[0], 0.1, 'n = ' + str(y.shape[1]) + ', F = ' + str(Fvalue) + ', p = ' + str(pvalue))

    # Add the extinction memory test data to the plot
    mean_response_to_CS_during_extinction_memory_test *= 100
    x = np.arange(mean_response_to_CS_during_extinction_memory_test.shape[1]) + offset
    plt.errorbar(x, np.mean(mean_response_to_CS_during_extinction_memory_test, 0), yerr=sem(mean_response_to_CS_during_extinction_memory_test, 0, ddof=0), color='k', linestyle='', marker='o', capsize=4, zorder=0)

    y = np.transpose(mean_response_to_CS_during_extinction_memory_test, [1, 0])

    pvalue, Fvalue = statsfunctions.repeated_measures_ANOVA(x, y)
    Fvalue = statsfunctions.display_n_sig_figs(Fvalue, 2)
    pvalue = statsfunctions.display_n_sig_figs(pvalue, 1)
    plt.text(x[0], 0.1, 'n = ' + str(y.shape[1]) + ', F = ' + str(Fvalue) + ', p = ' + str(pvalue))

    plt.xticks([1, 8,
                13 + 1, 13 + 5, 13 + 10, 13 + 15, 13 + 20, 13 + 25, 13 + 30,
                48 + 1, 48 + 5, 48 + 10, 48 + 15, 48 + 20, 48 + 25, 48 + 30,
                ],
               ['1', '8', '1', '5', '10', '15', '20', '25', '30', '1', '5', '10', '15', '20', '25', '30'])
    prettyplot.no_box()
    plt.ylim([-10, 10])

    z = mean_response_to_CS_during_conditioning
    if 'mPFC' in data_directory_path_day_1:
        mean_response_to_CS_during_conditioning = mean_response_to_CS_during_conditioning[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]  # There's one mouse that didn't get run on the last day. Exclude it for pairwise t-tests
        mean_response_to_CS_during_extinction = mean_response_to_CS_during_extinction[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]

    # print(maxima_tone1.shape, maxima_tone_test.shape, maxima_tone_test_day_2.shape)
    # firsts_and_lasts = np.array([maxima_tone1[:, 0], maxima_tone1[:, -1], maxima_tone_test[:, 0], maxima_tone_test[:, -1], maxima_tone_test_day_2[:, 0], maxima_tone_test_day_2[:, -1]])
    # t_tests = []
    # for i in range(firsts_and_lasts.shape[0] - 1):
    #     stats = ttest_rel(firsts_and_lasts[i, :], firsts_and_lasts[i + 1, :])
    #     t_tests.append(stats.pvalue)

    stats = ttest_rel(mean_response_to_NCS_during_conditioning[:, 0], mean_response_to_NCS_during_conditioning[:, -1])
    print('tone 2 p-value =', stats.pvalue, 'full stats =', stats)

    stats = ttest_rel(z[:, -1], mean_response_to_NCS_during_conditioning[:, -1])
    print('tone 1 vs. tone 2 p-value =', stats.pvalue, 'full stats =', stats)

    # print(t_tests)
    #
    # corrected_pvalues = mult_test(t_tests, method='bonferroni')[1]
    #
    # print(corrected_pvalues)

    plt.savefig(header + 'Average_CS_Response_Summary.pdf', transparent=True)
    plt.show()


data_directory_path_day_1 = '../../OriginalData/FearConditioningData/Day1/mPFC/'
data_directory_path_day_2 = '../../OriginalData/FearConditioningData/Day2/mPFC/'

DataFiles = {
    'mPFC': {
        'Brain Region': 'mPFC',
        'Day 1 Path': 'FearConditioningData/Day1/mPFC/',
        'Day 2 Path': 'FearConditioningData/Day2/mPFC/',
        'CS Name Day 2': 'tone_test', # The event name for the CS during extinction memory was renamed for the NAc
    },
    'NAcCore': {
        'Brain Region': 'NAcCore',
        'Day 1 Path': 'FearConditioningData/Day1/NAcCore/',
        'Day 2 Path': 'FearConditioningData/Day2/NAcCore/',
        'CS Name Day 2': 'tone1', # The event name for the CS during extinction memory was renamed for the NAc
    },
}

for key in DataFiles.keys():

    datafile = DataFiles[key]

    perform_analysis(datafile['Day 1 Path'], datafile['Day 2 Path'], datafile['Brain Region'], datafile['CS Name Day 2'])




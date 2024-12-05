import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import utils.statsfunctions as statsfunctions
import scipy.signal as signal
from scipy.stats import sem
from scipy.stats import ttest_rel


def perform_analysis(experiment_day_1, experiment_day_2):

    brain_region = experiment_day_1['Brain Region']
    CS_name_day_2 = 'CS (Unpaired)'
    header = 'FigurePdfs/IsosbesticPlots/FearConditioning_' + brain_region + '_'

    data_day_1 = experiment_day_1['data']
    data_day_2 = experiment_day_2['data']

    cue_duration = 5
    shock_duration = 0.5
    window = [-20, 20]
    averaging_window = [0.0, 4.5]


    def collect_trial_windows(data_list, event, window):

        window_list = [] * len(data_list)

        fs = data_list[0]['fs']
        index_window = [int(np.round(window[0] * fs)), int(np.round(window[1] * fs))]

        for data in data_list:
            within_data_windows = []
            events = data[event]

            for i in range(len(events)):

                t_event = events[i]

                index_event = np.argmin(np.square(data['times'] - t_event))
                within_data_windows.append(data['dF/F Isosbestic'][index_event + index_window[0]:index_event + index_window[1]])
                ts = data['times'][index_event + index_window[0]:index_event + index_window[1]] - data['times'][index_event]

            window_list.append(within_data_windows)

        window_list = np.array(window_list)
        sos = signal.bessel(4, 2, 'low', fs=fs, output='sos')
        window_list = signal.sosfiltfilt(sos, window_list, axis=2)

        return ts, window_list


    # Compute the average response to the CS during conditioning

    ts, tone1_window_list = collect_trial_windows(data_day_1, 'CS (Paired)', window)
    # baseline the data
    tone1_window_list = tone1_window_list - np.mean(tone1_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_conditioning = np.mean(tone1_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    # Compute the average response to the CS during extinction

    ts, tone_test_window_list = collect_trial_windows(data_day_1, 'CS (Unpaired)', window)
    # baseline the data
    tone_test_window_list = tone_test_window_list - np.mean(tone_test_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_extinction = np.mean(tone_test_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    # DAY 1 PERIEVENT DATA FOR CS/US DURING CONDITIONING


    # DAY 1 PERIEVENT DATA FOR NCS DURING CONDITIONING

    ts, tone2_window_list = collect_trial_windows(data_day_1, 'NCS', window)
    # baseline the data
    tone2_window_list = tone2_window_list - np.mean(tone2_window_list[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_NCS_during_conditioning = np.mean(tone2_window_list[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

    # DAY 2 PERIEVENT DATA

    ts_day_2, tone_test_window_list_day_2 = collect_trial_windows(data_day_2, CS_name_day_2, window)
    # Baseline the data
    tone_test_window_list_day_2 = tone_test_window_list_day_2 - np.mean(tone_test_window_list_day_2[:, :, ts < 0], axis=2, keepdims=True)
    # mean computation
    mean_response_to_CS_during_extinction_memory_test = np.mean(tone_test_window_list_day_2[:, :, np.logical_and(ts >= averaging_window[0], ts <= averaging_window[1])], axis=2)

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

    stats = ttest_rel(mean_response_to_NCS_during_conditioning[:, 0], mean_response_to_NCS_during_conditioning[:, -1])
    print('tone 2 p-value =', stats.pvalue, 'full stats =', stats)

    stats = ttest_rel(z[:, -1], mean_response_to_NCS_during_conditioning[:, -1])
    print('tone 1 vs. tone 2 p-value =', stats.pvalue, 'full stats =', stats)

    stats = ttest_rel(np.mean(mean_response_to_CS_during_extinction[:, [0, 1, 2, 3, 4]], 1), np.mean(mean_response_to_CS_during_extinction[:, [-5, -4, -3, -2, -1]], 1))
    print('CS at start of extinction vs. end', stats.pvalue, 'full stats =', stats)
    print('% dF/F before: ', np.mean(mean_response_to_CS_during_extinction[:, [0, 1, 2, 3, 4]]))
    print('% dF/F after: ', np.mean(mean_response_to_CS_during_extinction[:, [-5, -4, -3, -2, -1]]))

    stats = ttest_rel(np.mean(mean_response_to_CS_during_extinction_memory_test[:, [0, 1, 2, 3, 4]], 1), np.mean(mean_response_to_CS_during_extinction_memory_test[:, [-5, -4, -3, -2, -1]], 1))
    print('CS at start of extinction memory test vs. end', stats.pvalue, 'full stats =', stats)
    print('% dF/F before: ', np.mean(mean_response_to_CS_during_extinction_memory_test[:, [0, 1, 2, 3, 4]]))
    print('% dF/F after: ', np.mean(mean_response_to_CS_during_extinction_memory_test[:, [-5, -4, -3, -2, -1]]))

    plt.savefig(header + 'Average_CS_Response_Summary_Isosbestic.pdf', transparent=True)
    plt.show()


with open('../PreprocessedData/CompleteFearConditioningDataset.obj', "rb") as input_file:
    experiments = pickle.load(input_file)

perform_analysis(experiments['FC_mPFC_Day_1'], experiments['FC_mPFC_Day_2'])
perform_analysis(experiments['FC_NAcCore_Day_1'], experiments['FC_NAcCore_Day_2'])





# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# This file contains functions for gathering windows of data, plotting and handles some of the statistics.

import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import scipy.signal as signal
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests as mult_test

# Global constants
carousel_turning_time = 0.8                             # seconds to turn the carousel into position

# These indices are used in the metadata associated with the peri-event windows
ANIMAL_NUMBER = 0
TRIAL_NUMBER = 1
PORT = 2
QUININE_TRIAL_NUMBER = 3
TOTAL_LICK_NUMBER_OF_LICKS = 4
EVENT_TIME = 5

def baseline_signal(ts, data, window):
    """
    Mean subtract a signal over the specified window
    :param ts: array of times for each point in data
    :param data: the data to be baselined
    :param window: the window [t_start, t_end]
    :return: baselined signal
    """
    return data - np.mean(data[np.logical_and(ts >= window[0], ts <= window[1])])


def map_port_numbers_to_dose(array):
    """
    Fixes the raw index numbers of the port array (from the GPIO pins on the PI).
    Specifically, [13, 19, 6, 26, 5] becomes [0, 1, 2, 3, 4].

    :param array: Raw numbers
    :return: 0 for water, 1, 2, 3, 4 for the four quinine concentrations
    """
    quinine_dose_numbers = [13, 19, 6, 26, 5]  # 13 is water

    if type(array) == list:
        array = np.array(array, dtype=int)
    I = []
    for i in range(len(quinine_dose_numbers)):
        indices = np.where(np.equal(array, quinine_dose_numbers[i]))
        I.append(indices[0])

    for i in range(len(I)):
        np.put(array, I[i], i)

    return array


def get_latencies(metadata):
    number_of_animals = int(np.max(metadata[:, ANIMAL_NUMBER]))

    per_animal_latencies = []

    for animal_number in range(number_of_animals):
        I = metadata[:, ANIMAL_NUMBER] == animal_number

        md = metadata[I, :]

        # make sure that the trials are sorted
        I = np.argsort(md[:, TRIAL_NUMBER])
        md = md[I, :]

        latencies = [[], [], [], [], []]

        for i in range(md.shape[0] - 1):
            port_number = int(md[i, PORT])
            next_port_number = int(md[i + 1, PORT])
            latency = md[i + 1, EVENT_TIME] - md[i, EVENT_TIME]
            latencies[port_number].append(latency)


        for i in range(len(latencies)):
            latencies[i] = np.mean(latencies[i])

        latencies = np.array(latencies)
        per_animal_latencies.append(latencies)

    return np.transpose(np.array(per_animal_latencies), [1, 0])


def get_window(data, index, window_size):
    """
    Grabs a window of data:

    data[index + window_size[0]:(index + window_size[1] + 1)]
    """
    if index + window_size[0] < 0 or index + window_size[1] > len(data):
        raise Exception('Window out of bounds')

    return data[index + window_size[0]:(index + window_size[1] + 1)]


def filter_and_z_score_data(ts, data, lowpass_frequency=3, filter_order=4, zscore=True):
    """
    Filter and z score the data. Note that only times after 1 second are included in the std computation to remove artifacts when the recording starts.

    :param ts: the time of each datapoint (only used to compute the sampling rate)
    :param data: the signal
    :return: filtered and z-scored data
    """
    # Low pass the signal
    sos = signal.bessel(filter_order, lowpass_frequency, btype='low', fs=1 / (ts[1] - ts[0]), output='sos')
    data = signal.sosfiltfilt(sos, data)
    if zscore:
        z = (data - np.mean(data))/np.std(data[ts > 1])
        return z
    else:
        return data


def plot_dataset(data, xlim=[290, 690], solutionChar='Q', zscore=False, plot_isosbestic=False):
    fig = plt.figure(figsize=(10, 2))
    fig.subplots_adjust(bottom=0.25, left=0.1)

    if plot_isosbestic:
        Fluorescence = data['dF/F Isosbestic']
    else:
        Fluorescence = data['dF/F Excitation after isosbestic subtraction']
    ts = data['times']

    z = filter_and_z_score_data(ts, Fluorescence, zscore=zscore)

    # Convert from dF/F to %dF/F if not z-scoring
    if not zscore:
        z = 100*z

    first_lick_time = data['first_lick_time']
    port = data['port']

    indices_in_time_window = np.logical_and(ts >= xlim[0], ts <= xlim[1])
    plt.plot(ts[indices_in_time_window], z[indices_in_time_window])
    mx = np.max(z[indices_in_time_window])
    mn = np.min(z[indices_in_time_window])
    ylim = [0.5*(mx + mn) - 0.5*(mx - mn)*1.1, 0.5*(mx + mn) + 0.5*(mx - mn)*1.1]

    prettyplot.no_box()
    if zscore:
        prettyplot.ylabel('z-score')
    else:
        prettyplot.ylabel('% dF/F')
    prettyplot.xlabel('time s')

    for i in range(len(first_lick_time)):
        t = first_lick_time[i]
        if xlim[0] <= t <= xlim[1]:
            if port[i] == 0:
                plt.axvline(t, color=prettyplot.colors['blue'])
                plt.text(t, -2, 'W')

            else:
                plt.axvline(t, color=prettyplot.colors['red'])
                plt.text(t, -2, solutionChar + str(port[i]))

    for i in range(len(data['water_time'])):
        t = data['water_time'][i]
        if xlim[0] <= t <= xlim[1]:
            plt.axvspan(t - carousel_turning_time, t, color=[0.8, 0.8, 1])

    plt.xlim(xlim)
    plt.ylim(ylim)


def process_data(animal_number, data, window_size_s, event='first_lick_time', zscore=True, plot_isosbestic=False):

    if plot_isosbestic:
        Fluorescence = data['dF/F Isosbestic']
    else:
        Fluorescence = data['dF/F Excitation after isosbestic subtraction']
    window_size_n = [int(np.round(window_size_s[0] * data['fs'])), int(np.round(window_size_s[1] * data['fs']))]

    ts = data['times']

    # Filter and z score the data
    z = filter_and_z_score_data(ts, Fluorescence, zscore=zscore)

    event_time = np.array(data[event])
    if event == 'water_time':
        if len(event_time) > len(data['first_lick_time']):
            while event_time[1] < data['first_lick_time'][0]:
                event_time = event_time[1:]
            if event_time[-1] > data['first_lick_time'][-1]:
                event_time = event_time[:-1]

        event_time = event_time - 0.8 # This puts the event right at the crossing of the beam break, instead of the solonoid click

    if event_time.shape != np.array(data['first_lick_time']).shape:
        raise Exception('FearConditioningData size mismatch between first_lick_time ' + str(np.array(data['first_lick_time']).shape) + 'and event_time ' + str(event_time.shape))

    port = data['port']
    #port = map_port_numbers_to_dose(port)

    windows = []
    windows_that_didnt_fit = []
    for i in range(len(event_time)):
        try:
            windows.append(get_window(z, int(np.round(event_time[i] * data['fs'])), window_size_n))
        except Exception as e:
            windows_that_didnt_fit.append(i)

    window_ts = np.arange(len(windows[0])) / data['fs'] + window_size_s[0]

    quinine_trial_number = np.cumsum(port != 0)
    quinine_trial_number = quinine_trial_number - 1


    # Q contains the metadata for every trial
    metadata = np.ones(shape=(len(port), 6))
    metadata[:, 0] = animal_number             # Animal number
    metadata[:, 1] = np.arange(len(port))      # Trial number
    metadata[:, 2] = port                      # Port 0-4
    metadata[:, 3] = quinine_trial_number
    metadata[:, 4] = data['number of licks']
    metadata[:, 5] = event_time
    metadata = np.delete(metadata, windows_that_didnt_fit, axis=0)

    return metadata, np.array(windows), window_ts


def get_all_perievent_window_data(data, window, event_name='first_lick_time', zscore=True, plot_isosbestic=False):
    metadata = []
    windows = []
    for i, d in enumerate(data):
        m, w, window_ts = process_data(i, d, window, event_name, zscore=zscore, plot_isosbestic=plot_isosbestic)
        metadata.append(m)
        windows.append(w)
    windows = np.concatenate(windows, 0)
    metadata = np.concatenate(metadata, 0)

    return metadata, window_ts, windows

def get_within_animal_average(animal_number, metadata, windows):
    means = []
    sems = []
    I = metadata[:, ANIMAL_NUMBER] == animal_number
    ports = metadata[I, PORT]
    within_animal_windows = windows[I, :]
    for port in range(5):
        x = within_animal_windows[ports == port, :]
        means.append(np.mean(x, 0))
        sems.append(np.std(x, 0) / np.sqrt(x.shape[0]))

    return np.array(means), np.array(sems)

def multi_wilcoxon_test_bonferroni_corrected(data):
    """
    Perform a Wilcoxon sign rank test between every pair of the first index of data (the second index is replicates).
    Then correct the p-values with the Bonferroni correction

    :param data: a numpy array with axes (conditions, replicates)
    :return: the p-values and a list of the pairs (axis_i, axis_j)
    """
    pvals = []
    sign_ranks = []
    pairs = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            pairs.append((i, j))
            stats_data = stats.wilcoxon(data[i, :], data[j, :])
            pvals.append(stats_data.pvalue)
            sign_ranks.append(stats_data.statistic)
    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]


    return corrected_pvalues, pairs, sign_ranks

def multi_pairwise_t_test_bonferroni_corrected(data):
    """
    Perform a pairwise t-test between every pair of the first index of data (the second index is replicates).
    Then correct the p-values with the Bonferroni correction

    :param data: a numpy array with axes (conditions, replicates)
    :return: the p-values and a list of the pairs (axis_i, axis_j)
    """
    pvals = []
    pairs = []
    t_statistics = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            pairs.append((i, j))
            stats_data = stats.ttest_rel(data[i, :], data[j, :])
            pvals.append(stats_data.pvalue)
            t_statistics.append(stats_data.statistic)
    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]
    return corrected_pvalues, pairs, t_statistics

def multi_t_test_bonferroni_corrected(data):
    """
    Perform a pairwise t-test between every pair of the first index of data (the second index is replicates).
    Then correct the p-values with the Bonferroni correction

    :param data: a numpy array with axes (conditions, replicates)
    :return: the p-values and a list of the pairs (axis_i, axis_j)
    """
    pvals = []
    pairs = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            pairs.append((i, j))
            stats_data = stats.ttest_ind(data[i, :], data[j, :])
            pvals.append(stats_data.pvalue)
    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]

    return corrected_pvalues, pairs


def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    x = x[:, None] * np.ones(shape=y.shape)
    return stats.linregress(np.reshape(x, [-1]), np.reshape(y, [-1]))

# Round to the number of significant figures

def display_n_sig_figs(x, n):
    if np.abs(x) >= 0.001 and np.abs(x) < 10000:
        return np.format_float_positional(x, precision=n, unique=True, fractional=False, trim='k')
    else:
        return np.format_float_scientific(x, precision=n, unique=True)






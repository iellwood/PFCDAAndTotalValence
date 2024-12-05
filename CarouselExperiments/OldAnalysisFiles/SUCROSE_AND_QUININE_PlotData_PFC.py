import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import os
import scipy.signal as signal
import scipy.stats as stats
import DataProcessingFunctions


# Global constants
carousel_turning_time = 0.8                             # seconds to turn the carousel into position
quinine_concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]      # concentrations of quinine in mM
data_directory_path = '../../OriginalData/CarouselExperimentsData/SucroseAndQuinine/mPFC/'  # directory for data files
PFC_baseline_window = [-1, 0]                           # window for baselining the signal
perievent_window = [5, 10]
averaging_window = [0, 5]

# These indices are used in the metadata associated with the peri-event windows
ANIMAL_NUMBER = 0
TRIAL_NUMBER = 1
PORT = 2
QUININE_TRIAL_NUMBER = 3
TOTAL_LICK_NUMBER_OF_LICKS = 4


# Load the data
file_names = os.listdir(data_directory_path)
data = []
for file_name in file_names:
    with open(data_directory_path + file_name, "rb") as input_file:
        data.append(pickle.load(input_file))
        print('loaded file,', file_name)

number_of_animals = len(data)

fs = data[0]['fs']

# FIGURE PANEL: Plots an example dataset
if True:
    DataProcessingFunctions.plot_dataset(data[0], xlim=[290, 690], ylim=[-2.5, 8], solutionChar='S')
    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_ExampleDataset.pdf', transparent=True)
    plt.show()

# Collects all the windows from all the animals. The metadata tells the animal number and port for everyone window
metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(data, perievent_window)

# FIGURE PANEL: plot the lick rate as a function of quinine concentration
if True:
    lick_counts = []
    for port_number in range(5):
        average_number_of_licks = []
        for animal_number in range(number_of_animals):
            I = np.logical_and(metadata[:, ANIMAL_NUMBER] == animal_number, metadata[:, PORT] == port_number)

            average_number_of_licks.append(np.mean(metadata[I, TOTAL_LICK_NUMBER_OF_LICKS]))

        lick_counts.append(average_number_of_licks)

    lick_counts = np.array(lick_counts)
    np.save('SUCROSE_AND_QUININE_PFC_lick_counts', lick_counts)
    fig = plt.figure(figsize=(2, 4))
    plt.subplots_adjust(left=0.25, bottom=0.25)


    plt.errorbar(quinine_concentrations, np.mean(lick_counts, 1), yerr=np.std(lick_counts, 1)/np.sqrt(lick_counts.shape[1]), capsize=5)
    prettyplot.no_box()
    #plt.ylim([1, 6])
    plt.xticks(quinine_concentrations)
    prettyplot.ylabel('number of licks')
    prettyplot.xlabel('quinine concentration mM')
    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_NumberOfLicksVsQuinineConcentration.pdf', transparent=True)

    plt.show()

#FIGURE PANEL: Plots the latencies for each port.
# Note that we have not included water as after every water trial there is a 50% or 100% of a quinine trial for the first/second water trial
# In contrast, for quinine trials, they are always followed by water trials. Thus we cannot compare these latencies.
# if False:
#     fig = plt.figure(figsize=(2, 4))
#     latencies = DataProcessingFunctions.get_latencies(metadata)
#     plt.subplots_adjust(left=0.25, bottom=0.25)
#     plt.errorbar(quinine_concentrations, np.mean(latencies, 1), yerr=(np.std(latencies, 1)/np.sqrt(latencies.shape[1])), capsize=5)
#     prettyplot.no_box()
#     plt.ylim([0, 40])
#     plt.xticks(quinine_concentrations)
#     prettyplot.ylabel('latency s')
#     prettyplot.xlabel('quinine concentration mM')
#     plt.show()

# SUPPLEMENTARY FIGURE PANEL: Plots the peri-event averages and single-trial responses from one animal
animal_number_for_plot = 3
if False:
    fig, axes = plt.subplots(1, 5, figsize=(4, 4))
    fig.subplots_adjust(bottom=0.25, left=0.25)
    for port_number in range(5):
        axis = axes[port_number]
        if port_number == 0:
            prettyplot.no_box(axis)
            prettyplot.ylabel('GRABDA3h dF/F, z-score')
        else:
            prettyplot.x_axis_only(axis)
        I = np.logical_and(metadata[:, PORT] == port_number, metadata[:, ANIMAL_NUMBER] == animal_number_for_plot)
        w = windows[I, :]

        for k in range(w.shape[0]):
            if port_number == 0 and k >= 10:
                break
            m = w[k, :]
            m = m - np.mean(m[np.logical_and(window_ts <= 0, window_ts >= -1)])
            axis.plot(window_ts, m, color=[0.75, 0.75, 0.75])
            axis.set_ylim([-1, 7])

        m = np.mean(w, 0)
        m = m - np.mean(m[np.logical_and(window_ts <= 0, window_ts >= -1)])
        axis.plot(window_ts, m, color='k')

    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_SingleAnimalTrials.pdf', transparent=True)
    plt.show()


# In this test, the height of the phasic release event is compared with the baseline height.
# Have played around with different port numbers and animal numbers, but no correlation was found.
# Not included in paper
# if False:
#     I = metadata[:, ANIMAL_NUMBER] == 7
#
#     ws = windows[I, :]
#
#     mins = np.mean(ws[:, np.logical_and(window_ts <= 0.5, window_ts >= -0.5)], 1)
#     maxs = np.mean(ws[:, np.logical_and(window_ts <= 2.5, window_ts >= 1.5)], 1)
#
#     for i in range(ws.shape[0]):
#         plt.plot(window_ts, ws[i, :])
#         plt.plot([0, 2], [mins[i], maxs[i]])
#
#     plt.show()
#
#     plt.scatter(mins, maxs - mins)
#     plt.show()


means = []
sems = []
for i in range(number_of_animals):
    m, s = DataProcessingFunctions.get_within_animal_average(i, metadata, windows)
    means.append(m)
    sems.append(s)


means = np.array(means)
sems = np.array(sems)

#fig, axes = plt.subplots(10, 5)

mx = np.max(means)
mn = np.min(means)
mid = (mx + mn)*0.5
dy = (mx - mn)*0.5

integrated_DA = np.zeros(shape=(number_of_animals, 5))

for animal_number in range(number_of_animals):
    for concentration in range(5):
        m = means[animal_number, concentration]
        m = m - np.mean(m[np.logical_and(window_ts >= PFC_baseline_window[0], window_ts <= PFC_baseline_window[1])])
        sem = sems[animal_number, concentration]

        integrated_DA[animal_number, concentration] = np.mean(m[np.logical_and(window_ts >= averaging_window[0], window_ts < averaging_window[1])])

# FIGURE PANEL: Plots the size of the dopamine response vs. the concentration of quinine
if True:
    fig = plt.figure(figsize=(3, 4))
    fig.add_axes([0.2, 0.2, .6, .7])


    for animal_number in range(number_of_animals):
        plt.scatter(quinine_concentrations, integrated_DA[animal_number, :], color=[0.75, 0.75, 0.75], marker='o')

    integrated_DA_NAcCore = np.load('SUCROSE_AND_QUININE_NAc_DA_vs_Quinine_SummaryData.npy')

    plt.errorbar(quinine_concentrations, np.mean(integrated_DA_NAcCore, 0),
                 yerr=np.std(integrated_DA_NAcCore, 0) / np.sqrt(integrated_DA_NAcCore.shape[0]), color=prettyplot.colors['blue'], marker='o', capsize=5)


    plt.errorbar(quinine_concentrations, np.mean(integrated_DA, 0),
                 yerr=np.std(integrated_DA, 0) / np.sqrt(integrated_DA.shape[0]), color='k', marker='o', capsize=5)

    plt.axhline(0, color='k')

    plt.xticks(quinine_concentrations)
    prettyplot.no_box()
    prettyplot.xlabel("Quinine conc. mM")
    prettyplot.ylabel("GRABDA3h dF/F, z-score")

    pvalues, pairs = DataProcessingFunctions.multi_pairwise_t_test_bonferroni_corrected(np.transpose(integrated_DA, [1, 0]))
    y = np.transpose(integrated_DA, [1, 0])
    print('y shape =', y.shape)

    print('Statistics for Quinine summary plot (mPFC)')
    print('ANOVA p-value =', stats.f_oneway(y[0, :], y[1, :], y[2, :], y[3, :], y[4, :]))

    Qs = ['W', 'Q1', 'Q2', 'Q3', 'Q4']
    for i in range(len(pvalues)):
        print('Pair', Qs[pairs[i][0]], 'vs.', Qs[pairs[i][1]], 'pvalue =', pvalues[i])

    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_FVsQuinineConcentrationSummary.pdf', transparent=True)
    plt.show()

# FIGURE PANEL: Plots the peri-event average across all animals of dopamine vs. time
if True:
    fig, axes = plt.subplots(1, 5, figsize=(8, 2))
    fig.subplots_adjust(bottom=0.25, left=0.25)
    for i in range(5):
        axis = axes[i]
        m = np.mean(means[:, i, :], 0)
        m = m - np.mean(m[np.logical_and(window_ts >= -2, window_ts <= -1)])
        sem = np.std(means[:, i, :], 0)/np.sqrt(means[:, i, :].shape[0])
        print(sem.shape)
        axis.fill_between(window_ts, m - sem, m + sem, color=[0.8, 0.8, 0.8])
        axis.plot(window_ts, m, color='k')

        if i == 0:
            prettyplot.no_box(axis)
            prettyplot.ylabel('z-score', axis=axis)
            prettyplot.xlabel('time s', axis=axis)
        else:
            prettyplot.x_axis_only(axis)

        axis.set_ylim([-0.25, 1])
    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_AverageResponseVsQuinineConcentration.pdf', transparent=True)

    plt.show()


def get_average_first_n_blocks(block_size, number_of_blocks, metadata, windows, port=None):
    means = []
    sems = []
    for block_number in range(number_of_blocks):
        windows_averaged_across_animals = []
        for animal_number in range(number_of_animals):

            # Begin by checking that there are enough trials in this animal to cover all the blocks
            Z = metadata[:, ANIMAL_NUMBER] == animal_number
            if port is not None:
                Z = np.logical_and(Z, metadata[:, PORT] > 0)
            if np.sum(Z) >= block_size * number_of_blocks: # Test if there are enough blocks

                I = np.logical_and(block_number * block_size <= metadata[:, QUININE_TRIAL_NUMBER], (block_number + 1) * block_size > metadata[:, QUININE_TRIAL_NUMBER])
                if port is not None:
                    J = metadata[:, PORT] == port
                else:
                    J = metadata[:, PORT] > 0
                I = np.logical_and(J, I)
                I = np.logical_and(I, metadata[:, ANIMAL_NUMBER] == animal_number)
                if np.sum(I) > 0:
                    w = windows[I, :]
                    windows_averaged_across_animals.append(np.mean(w, 0))

        windows_averaged_across_animals = np.array(windows_averaged_across_animals)
        means.append(np.mean(windows_averaged_across_animals, 0))
        sems.append(np.std(windows_averaged_across_animals, 0)/np.sqrt(windows_averaged_across_animals.shape[0]))

    return np.array(means), np.array(sems)

# SUPP. FIGURE PANEL: Plots dopamine vs. time across all animals in blocks of 4 trials (to see time dependence of the quinine response)
if False:
    block_size = 4
    number_of_blocks = 6
    fig, axes = plt.subplots(number_of_blocks, 5)
    for port_number in range(0, 5):
        means, sems = get_average_first_n_blocks(block_size, number_of_blocks, metadata=metadata, windows=windows, port=port_number)
        for i in range(number_of_blocks):
            axis = axes[i, port_number]
            m = means[i, :]
            m = m - np.mean(m[np.logical_and(window_ts <= 0, window_ts >= -1)])
            sem = sems[i, :]
            axis.fill_between(window_ts, m - sem, m + sem, color=[0.8, 0.8, 1])
            axis.plot(window_ts, m)

            axis.set_ylim([-0.25, 3])
            if port_number > 0 and i == number_of_blocks - 1:
                prettyplot.x_axis_only(axis)
            elif port_number == 0 and i == number_of_blocks - 1:
                prettyplot.no_box(axis)
            elif port_number ==0 and i < number_of_blocks - 1:
                prettyplot.y_axis_only(axis)
            else:
                prettyplot.no_axis(axis)

    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_ReleaseVsBlockNumber.pdf', transparent=True)
    plt.show()


# Peri-event average around the beam-break

metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(data, [5, 10], event_name='water_time')

print('metadata.shape =', metadata.shape)
print('windows.shape =', windows.shape)

means = []
sems = []
for i in range(number_of_animals):
    m, s = DataProcessingFunctions.get_within_animal_average(i, metadata, windows)
    means.append(m)
    sems.append(s)

means = np.array(means)
sems = np.array(sems)
mx = np.max(means)
mn = np.min(means)
mid = (mx + mn)*0.5
dy = (mx - mn)*0.5

if False:
    fig, axes = plt.subplots(1, 5, figsize=(8, 2))
    fig.subplots_adjust(bottom=0.25, left=0.25)
    for i in range(5):
        axis = axes[i]
        m = np.mean(means[:, i, :], 0)
        m = m - np.mean(m[np.logical_and(window_ts <= 0, window_ts >= -1)])
        sem = np.std(means[:, i, :], 0)/np.sqrt(means[:, i, :].shape[0])
        print(sem.shape)
        axis.fill_between(window_ts, m - sem, m + sem, color=[0.8, 0.8, 1])
        axis.plot(window_ts, m)

        if i == 0:
            prettyplot.no_box(axis)
            prettyplot.ylabel('z-score', axis=axis)
            prettyplot.xlabel('time s', axis=axis)
        else:
            prettyplot.x_axis_only(axis)

        axis.set_ylim([-0.25, 1.75])
    plt.savefig('FigurePdfs/SUCROSE_AND_QUININE_PFC_PerieventAverage_BeamBreak.pdf', transparent=True)

    plt.show()

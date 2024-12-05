import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import os
import scipy.signal as signal
import scipy.stats as stats
import DataProcessingFunctions
import utils.statsfunctions as statsfunctions

def analyze_dataset(
        data_directory_path,
        experiment_name,
        brain_region,
        concentrations,
        solute,
        perievent_window,
        baseline_window,
        averaging_window,
        example_dataset=-1,
        example_dataset_xlim=None,
        zscore_data=False,
        ylim=[-0.5, 2],
        plot_non_iso=True,
):
    header = experiment_name + '_' + brain_region + '_'

    # Global constants

    # These indices are used in the metadata associated with the peri-event windows
    ANIMAL_NUMBER = 0
    TRIAL_NUMBER = 1
    PORT = 2
    QUININE_TRIAL_NUMBER = 3
    TOTAL_LICK_NUMBER_OF_LICKS = 4

    if zscore_data:
        multiplicative_factor = 1
    else:
        multiplicative_factor = 100

    # Load the data
    file_names = os.listdir(data_directory_path)
    data = []
    for file_name in file_names:
        with open(data_directory_path + file_name, "rb") as input_file:
            data.append(pickle.load(input_file))

    number_of_animals = len(data)

    for key in data[0].keys():
        print(key)
    exit()

    # dataset_number = 2
    # fs = data[dataset_number]['fs']
    # ts = np.arange(data[dataset_number]['F_ex'].shape[0])/data[0]['fs']
    # plt.plot(ts, data[dataset_number]['F_ex'], color=prettyplot.colors['black'])
    # plt.plot(ts, data[dataset_number]['F_iso'], color=prettyplot.colors['blue'])
    # plt.show()

    # FIGURE PANEL: Plots an example dataset
    # if example_dataset >= 0:
    #     DataProcessingFunctions.plot_dataset(data[example_dataset], xlim=example_dataset_xlim, zscore=False, plot_isosbestic=True)
    #     plt.savefig('FigurePdfs/IsosbesticFigures/' + header + 'ExampleDataset.pdf', transparent=True)
    #     plt.show()

    # Collects all the windows from all the animals. The metadata tells the animal number and port for everyone window
    metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(data, perievent_window, zscore=zscore_data, plot_isosbestic=False)
    metadata_iso, window_ts_iso, windows_iso = DataProcessingFunctions.get_all_perievent_window_data(data, perievent_window, zscore=zscore_data, plot_isosbestic=True)

    # Average within animal
    means = []
    sems = []
    for i in range(number_of_animals):
        m, s = DataProcessingFunctions.get_within_animal_average(i, metadata, windows)
        means.append(m)
        sems.append(s)
    means = np.array(means)

    # Average within animal
    means_iso = []
    sems_iso = []
    for i in range(number_of_animals):
        m, s = DataProcessingFunctions.get_within_animal_average(i, metadata, windows_iso)
        means_iso.append(m)
        sems_iso.append(s)
    means_iso = np.array(means_iso)


    # FIGURE PANEL: Plots the peri-event average across all animals of fluorescence vs. time
    if True:
        fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.5))
        fig.subplots_adjust(bottom=0.25, left=0.25)
        for i in range(5):
            axis = axes[i]

            if plot_non_iso:
                m = np.mean(means[:, i, :], 0)
                m = m - np.mean(m[np.logical_and(window_ts <= baseline_window[1], window_ts >= baseline_window[0])])
                sem = np.std(means[:, i, :], 0)/np.sqrt(means[:, i, :].shape[0])
                axis.fill_between(window_ts, multiplicative_factor * (m - sem), multiplicative_factor * (m + sem), color=[0.8, 0.8, 0.8])
                axis.plot(window_ts, multiplicative_factor * m, color='k')


            m = np.mean(means_iso[:, i, :], 0)
            m = m - np.mean(m[np.logical_and(window_ts_iso <= baseline_window[1], window_ts_iso >= baseline_window[0])])
            sem = np.std(means_iso[:, i, :], 0)/np.sqrt(means_iso[:, i, :].shape[0])
            axis.fill_between(window_ts_iso, multiplicative_factor * (m - sem), multiplicative_factor * (m + sem), color=[0.8, 0.8, 1])
            axis.plot(window_ts, multiplicative_factor * m, color=prettyplot.colors['blue'])

            if i == 0:
                prettyplot.no_box(axis)
                if zscore_data:
                    prettyplot.ylabel('z-score', axis=axis)
                else:
                    prettyplot.ylabel('%dF/F', axis=axis)
                prettyplot.xlabel('time s', axis=axis)

            else:
                prettyplot.x_axis_only(axis)

        print('ylim =', ylim)
        for i in range(5):
            axes[i].set_ylim(ylim)

        plt.savefig('FigurePdfs/IsosbesticFigures/' + header + 'PerieventAverage_vs_Concentration.pdf', transparent=True)

        plt.show()


concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]
data_directory_path = '../../OriginalData/CarouselExperimentsData/Quinine/NAcCore/'
baseline_window = [-5, -2.5]
perievent_window = [5, 20]
averaging_window = [0, 5]

quinine_concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]
sucrose_concentrations = [0.0, 29.0, 58.0, 117.0, 233.0]      # concentrations of quinine in mM

datasets = {

    # Quinine Experiments

    'Quinine mPFC': {
         'Experiment Name': 'CarouselExperiment_Quinine',
         'Brain Region': 'mPFC',
         'Path': 'FearConditioningData/Quinine/mPFC/',
         'Solute': 'quinine',
         'Solute Concentrations': quinine_concentrations,
         'Baseline Window': [-5, -2.5],
         'Example dataset': 10,
         'Example dataset xlim': [290, 690],
         'ylim': [-1, 2],
         'plot_non_iso': True,
    },

    'Quinine NAcCore': {
         'Experiment Name': 'CarouselExperiment_Quinine',
         'Brain Region': 'NAc_Core',
         'Path': 'FearConditioningData/Quinine/NAcCore/',
         'Solute': 'quinine',
         'Solute Concentrations': quinine_concentrations,
         'Baseline Window': [-5, -2.5],
         'Example dataset': -1,
         'Example dataset xlim': None,
         'ylim': [-5, 15],
         'plot_non_iso': True,
     },

    # SUCROSE EXPERIMENTS

    'Sucrose mPFC': {
         'Experiment Name': 'CarouselExperiment_Sucrose',
         'Brain Region': 'mPFC',
         'Path': 'FearConditioningData/Sucrose/mPFC/',
         'Solute': 'sucrose',
         'Solute Concentrations': sucrose_concentrations,
         'Baseline Window': [-10, -5],
         'Example dataset': -1,
         'Example dataset xlim': None,
         'ylim': [-1, 2],
         'plot_non_iso': False,
     },

    'Sucrose NAcCore': {
         'Experiment Name': 'CarouselExperiment_Sucrose',
         'Brain Region': 'NAc_Core',
         'Path': 'FearConditioningData/Sucrose/NAcCore/',
         'Solute': 'sucrose',
         'Solute Concentrations': sucrose_concentrations,
         'Baseline Window': [-10, -5],
         'Example dataset': -1,
         'Example dataset xlim': None,
         'ylim': [-5, 15],
         'plot_non_iso': False,
     },

    # SUCROSE + QUININE Experiments

    'SucroseAndQuinine mPFC': {
         'Experiment Name': 'CarouselExperiment_SucroseAndQuinine',
         'Brain Region': 'mPFC',
         'Path': 'FearConditioningData/SucroseAndQuinine/mPFC/',
         'Solute': 'quinine',
         'Solute Concentrations': quinine_concentrations,
         'Baseline Window': [-5, -2.5],
        'Example dataset': -1,
        'Example dataset xlim': None,
         'ylim': [-1, 2],
         'plot_non_iso': False,
     },

    'SucroseAndQuinine NAcCore': {
         'Experiment Name': 'CarouselExperiment_SucroseAndQuinine',
         'Brain Region': 'NAc_Core',
         'Path': 'FearConditioningData/SucroseAndQuinine/NAcCore/',
         'Solute': 'quinine',
         'Solute Concentrations': quinine_concentrations,
         'Baseline Window': [-10, -5],
         'Example dataset': -1,
         'Example dataset xlim': None,
         'ylim': [-5, 15],
         'plot_non_iso': False,
     },

}

for key in datasets.keys():
    dataset = datasets[key]
    print('Analysis of', dataset['Experiment Name'], '(' + dataset['Brain Region'] + ')')

    analyze_dataset(
        dataset['Path'],
        dataset['Experiment Name'],
        dataset['Brain Region'],
        dataset['Solute Concentrations'],
        dataset['Solute'],
        [10, 20], # perievent window, from -5 to +10
        dataset['Baseline Window'], # baseline window. The average of the F over this window is subtracted from the signal
        [0, 5],     # averaging window.
        example_dataset=dataset['Example dataset'],
        example_dataset_xlim=dataset['Example dataset xlim'],
        ylim=dataset['ylim'],
        plot_non_iso=dataset['plot_non_iso'],
    )
# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Produces the example plots of the large variability in mPFC dopamine responses to repeated stimuli

import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import DataProcessingFunctions

# These indices are used in the metadata associated with the peri-event windows
ANIMAL_NUMBER = 0
TRIAL_NUMBER = 1
PORT = 2
QUININE_TRIAL_NUMBER = 3
TOTAL_LICK_NUMBER_OF_LICKS = 4
EVENT_TIME = 5

def analyze_dataset(
        experiment_dictionary,
        perievent_window,
        plot_window,
        baseline_window,
        averaging_window,
        example_dataset=-1,
        example_dataset_xlim=None,
        zscore_data=True,
):


    header = experiment_dictionary['Experiment Name'] + '_' + experiment_dictionary['Brain Region'] + '_'

    # Global constants

    # These indices are used in the metadata associated with the peri-event windows
    ANIMAL_NUMBER = 0
    TRIAL_NUMBER = 1
    PORT = 2
    QUININE_TRIAL_NUMBER = 3
    TOTAL_NUMBER_OF_LICKS = 4

    solute = experiment_dictionary['Solute']
    concentrations = experiment_dictionary['Solute Concentrations']

    number_of_animals = len(experiment_dictionary['data'])

    # # FIGURE PANEL: Plots an example dataset
    # if example_dataset >= 0:
    #     DataProcessingFunctions.plot_dataset(experiment_dictionary['data'][example_dataset], xlim=example_dataset_xlim, zscore=False)
    #     plt.savefig('FigurePdfs/MainFigures/' + header + 'ExampleDataset.pdf', transparent=True)
    #     plt.show()

    # Collects all the windows from all the animals. The metadata tells the animal number and port for everyone window
    metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(experiment_dictionary['data'], perievent_window, zscore=False)

    windows = windows[:, np.logical_and(window_ts >= -1, window_ts <= 20)]
    window_ts = window_ts[np.logical_and(window_ts >= -1, window_ts <= 20)]

    windows = windows * 100

    for i in range(windows.shape[0]):
        windows[i, :] = windows[i, :] - np.mean(windows[i, :][np.logical_and(window_ts >= baseline_window[0], window_ts <= baseline_window[1])])

    print(window_ts.shape, windows.shape)

    example_animal_indices = metadata[:, ANIMAL_NUMBER] == example_dataset

    max_F = np.max(windows[example_animal_indices, :])
    min_F = np.min(windows[example_animal_indices, :])
    y_max = max_F + (max_F - min_F) * 0.05
    y_min = min_F - (max_F - min_F) * 0.05
    print('ylims=', [y_min, y_max])

    window_mask = np.logical_and(example_animal_indices, metadata[:, PORT] == 4)
    indices = np.arange(window_mask.shape[0])[window_mask]


    fig, axes = plt.subplots(1, indices.shape[0], figsize=(6.5, 2.5))
    fig.subplots_adjust(bottom=0.25, left=0.25)

    for i in range(len(axes)):
        index = indices[i]
        axis = axes[i]
        if i == 0:
            prettyplot.no_box(axis)
        else:
            prettyplot.x_axis_only(axis)

        trial_number = metadata[index, TRIAL_NUMBER]
        w = windows[index, :]
        axis.plot(window_ts, w, color='k')
        axis.set_xlim([-1, 20])
        axis.set_ylim([y_min, y_max])
        axis.text(0, y_max, 'trial ' + str(int(trial_number)))
    plt.savefig('FigurePdfs/TrialByTrialVariability/' + header + 'port_4_trial_dependence.pdf', transparent=True)
    plt.show()



# -------------------------------------------------------------------------------------------------------------------- #
# Note that we used different baseline windows for the mPFC and NAc data.
# -------------------------------------------------------------------------------------------------------------------- #
# This was motivated by
# 1) The mPFC dopamine signal has large baseline shifts from the effects of previous trials as the
# time constant of mPFC dopamine is slow. We thus selected a short interval immediately before the event.
# 2) The NAc dopamine signal appears to ramp up just before events. We thus selected a baseline interval a few seconds
# before the event

mPFC_baseline_window = [-1, 0]
NAc_baseline_window = [-15, -5]

datasets = {

    # Quinine Experiments
    'Carousel Experiment, Quinine, mPFC': { # Representative examples 2, 3, 11
         'Baseline Window': mPFC_baseline_window,
         'Example dataset': 3,
         'Example dataset xlim': [290, 690],
    },

    # SUCROSE EXPERIMENTS

    'Carousel Experiment, Sucrose, mPFC': {
         'Baseline Window': mPFC_baseline_window,
         'Example dataset': 5,
         'Example dataset xlim': None,
     },

}

with open('../PreprocessedData/CompleteCarouselDataset.obj', "rb") as input_file:
    experiments = pickle.load(input_file)

for key in datasets.keys():
    dataset = datasets[key]
    experiment = experiments[key]

    print('Analysis of', key, '(' + experiment['Brain Region'] + ')')

    analyze_dataset(
        experiment_dictionary=experiment,
        perievent_window=[-20, 20],
        plot_window = [-10, 20],
        baseline_window=dataset['Baseline Window'],
        averaging_window=[0, 5],
        example_dataset=dataset['Example dataset'],
        example_dataset_xlim=dataset['Example dataset xlim'],
    )
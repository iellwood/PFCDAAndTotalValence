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
        zscore_data=True,
):



    # Load the data
    file_names = os.listdir(data_directory_path)
    data = []
    for file_name in file_names:
        with open(data_directory_path + file_name, "rb") as input_file:
            data.append(pickle.load(input_file))

    number_of_animals = len(data)
    lick_averages = []
    for d in data:

        ports = DataProcessingFunctions.map_port_numbers_to_dose(d['port'])

        licks = d['lick_rate']
        lick_average = []
        for i in range(5):
            I = ports == i
            lick_average.append(np.mean(licks[I]))


        lick_average = np.array(lick_average)

        lick_averages.append(lick_average)

    lick_averages = np.array(lick_averages)

    print(lick_averages)

    print(lick_averages.shape)

    plt.plot(np.mean(lick_averages, axis=0))
    plt.gca().set_ylim(bottom=1)
    plt.show()





baseline_window = [-5, -2.5]
perievent_window = [5, 20]
averaging_window = [0, 5]
quinine_concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]
sucrose_concentrations = [0.0, 29.0, 58.0, 117.0, 233.0]      # concentrations of quinine in mM

datasets = {

    # Quinine Experiments
    'Quinine mPFC Old': {
        'Experiment Name': 'CarouselExperiment_Quinine',
        'Brain Region': 'mPFC',
        'Path': 'FearConditioningData/Quinine_Old/',
        'Solute': 'quinine',
        'Solute Concentrations': quinine_concentrations,
        'Baseline Window': [-5, -2.5],
        'Example dataset': 10,
        'Example dataset xlim': [290, 690],
    },

    'Quinine mPFC': {
         'Experiment Name': 'CarouselExperiment_Quinine',
         'Brain Region': 'mPFC',
         'Path': 'FearConditioningData/Quinine/mPFC/',
         'Solute': 'quinine',
         'Solute Concentrations': quinine_concentrations,
         'Baseline Window': [-5, -2.5],
         'Example dataset': 10,
         'Example dataset xlim': [290, 690],
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
     },

}

for key in ['Quinine mPFC Old']:
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
    )
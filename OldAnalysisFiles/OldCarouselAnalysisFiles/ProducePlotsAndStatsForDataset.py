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
    header = experiment_name + '_' + brain_region + '_'

    # Global constants

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

    number_of_animals = len(data)

    fs = data[0]['fs']

    # FIGURE PANEL: Plots an example dataset
    if example_dataset >= 0:
        DataProcessingFunctions.plot_dataset(data[example_dataset], xlim=example_dataset_xlim, zscore=False)
        plt.savefig('FigurePdfs/MainFigures/' + header + 'ExampleDataset.pdf', transparent=True)
        plt.show()

    # Collects all the windows from all the animals. The metadata tells the animal number and port for everyone window
    metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(data, perievent_window, zscore=zscore_data)


    # Save the number of licks at each concentration
    lick_counts = []
    for port_number in range(5):
        average_number_of_licks = []
        for animal_number in range(number_of_animals):
            I = np.logical_and(metadata[:, ANIMAL_NUMBER] == animal_number, metadata[:, PORT] == port_number)

            average_number_of_licks.append(np.mean(metadata[I, TOTAL_LICK_NUMBER_OF_LICKS]))

        lick_counts.append(average_number_of_licks)

    lick_counts = np.array(lick_counts)
    np.save('LickData/' + header + 'lick_counts', lick_counts)

    # Average within animal
    means = []
    sems = []
    for i in range(number_of_animals):
        m, s = DataProcessingFunctions.get_within_animal_average(i, metadata, windows)
        means.append(m)
        sems.append(s)
    means = np.array(means)
    sems = np.array(sems)

    # Compute the mean fluorescence across the averaging window
    mean_F_across_averaging_window = np.zeros(shape=(number_of_animals, 5))
    for animal_number in range(number_of_animals):
        for concentration in range(5):
            m = means[animal_number, concentration]
            m = m - np.mean(m[np.logical_and(window_ts >= baseline_window[0], window_ts <= baseline_window[1])])
            sem = sems[animal_number, concentration]

            mean_F_across_averaging_window[animal_number, concentration] = np.mean(m[np.logical_and(window_ts >= averaging_window[0], window_ts < averaging_window[1])])

    # FIGURE PANEL: Plots the size of the dopamine response vs. the concentration
    if True:

        y = np.transpose(mean_F_across_averaging_window, [1, 0])
        pvalue, Fvalue = statsfunctions.repeated_measures_ANOVA(np.array(concentrations), y)

        # For pairwise statistics, all possible comparisons are made with a Bonferroni correction.
        # This is used in the text to argue that the lowest concentration of quinine increases fluorescence relative
        # to pure water.
        pvalues, pairs = DataProcessingFunctions.multi_pairwise_t_test_bonferroni_corrected(np.transpose(mean_F_across_averaging_window, [1, 0]))
        concentration_names = ['W', 'C1', 'C2', 'C3', 'C4']
        for i in range(len(pvalues)):
            if pvalues[i] <= 0.05:
                print('Significant Pair', concentration_names[pairs[i][0]], 'vs.', concentration_names[pairs[i][1]], 'pvalue =', pvalues[i])

        fig = plt.figure(figsize=(3, 4))
        plt.subplots_adjust(left=0.5)
        fig.add_axes([0.2, 0.2, .4, .4])

        if zscore_data:
            multiplicative_factor = 1
        else:
            multiplicative_factor = 100 # If not z-scoring, multiply everything by 100

        for animal_number in range(number_of_animals):
            plt.scatter(concentrations, multiplicative_factor * mean_F_across_averaging_window[animal_number, :], color=[0.75, 0.75, 0.75], marker='o')

        plt.errorbar(concentrations, multiplicative_factor * np.mean(mean_F_across_averaging_window, 0),
                     yerr=multiplicative_factor * np.std(mean_F_across_averaging_window, 0) / np.sqrt(mean_F_across_averaging_window.shape[0]), color='k', marker='o', capsize=5, linestyle='')

        plt.axhline(0, color='k')

        plt.xticks(concentrations)
        prettyplot.no_box()
        prettyplot.xlabel(solute + " conc. mM")
        if zscore_data:
            prettyplot.ylabel("GRABDA3h dF/F, z-score")
        else:
            prettyplot.ylabel("GRABDA3h %dF/F")

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        ymid = (ylim[0] + ylim[1]) * 0.5
        dy = (ylim[1] - ylim[0]) * 0.5
        text_y = ymid + dy*0.7

        Fvalue = DataProcessingFunctions.display_n_sig_figs(Fvalue, 2)
        pvalue = DataProcessingFunctions.display_n_sig_figs(pvalue, 2)
        plt.text(concentrations[0], text_y, 'n = ' + str(number_of_animals) + ', F = ' + str(Fvalue) + ', p = ' + str(pvalue))

        np.save('SummaryData/' + header + 'mean_F_across_averaging_window', mean_F_across_averaging_window)

        plt.savefig('FigurePdfs/MainFigures/' + header + 'F_vs_ConcentrationSummary.pdf', transparent=True)
        plt.show()

    # FIGURE PANEL: Plots the peri-event average across all animals of fluorescence vs. time
    if True:
        fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.5))
        fig.subplots_adjust(bottom=0.25, left=0.25)
        for i in range(5):
            axis = axes[i]
            m = np.mean(means[:, i, :], 0)
            m = m - np.mean(m[np.logical_and(window_ts <= baseline_window[1], window_ts >= baseline_window[0])])
            sem = np.std(means[:, i, :], 0)/np.sqrt(means[:, i, :].shape[0])

            axis.fill_between(window_ts, multiplicative_factor * (m - sem), multiplicative_factor * (m + sem), color=[0.8, 0.8, 0.8])
            axis.plot(window_ts, multiplicative_factor * m, color='k')

            if i == 0:
                prettyplot.no_box(axis)
                if zscore_data:
                    prettyplot.ylabel('z-score', axis=axis)
                else:
                    prettyplot.ylabel('%dF/F', axis=axis)
                prettyplot.xlabel('time s', axis=axis)

                ymax = multiplicative_factor * np.max(m + sem)
                ymin = multiplicative_factor * np.min(m - sem)

            else:
                prettyplot.x_axis_only(axis)
                ymax = np.maximum(ymax, multiplicative_factor * np.max(m + sem))
                ymin = np.minimum(ymin, multiplicative_factor * np.min(m - sem))


        ymid = (ymax + ymin)*0.5
        dy = ymax - ymid
        ylim = [ymid - dy*1.05, ymid + dy*1.05]

        for i in range(5):
            axes[i].set_ylim(ylim)

        plt.savefig('FigurePdfs/MainFigures/' + header + 'PerieventAverage_vs_Concentration.pdf', transparent=True)

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
    )
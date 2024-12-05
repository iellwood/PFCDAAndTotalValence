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
        experiment_dictionary,
        perievent_window,
        baseline_window,
        averaging_window,
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

    # Collects all the windows from all the animals. The metadata tells the animal number and port for everyone window
    metadata, window_ts, windows = DataProcessingFunctions.get_all_perievent_window_data(experiment_dictionary['data'], perievent_window, zscore=zscore_data)

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
        print('Repeated measures ANOVA', 'F =', Fvalue, 'p =', pvalue, 'n =', y.shape[1])

        # For pairwise statistics, all comparisons between water, the lowest and highest concentrations are made with a
        # Bonferroni correction.
        # This is used in the text to argue that the lowest concentration of quinine increases fluorescence relative
        # to pure water.

        pvalues, pairs, t_statistics = DataProcessingFunctions.multi_pairwise_t_test_bonferroni_corrected(np.transpose(mean_F_across_averaging_window, [1, 0])[[0, 1, 4], :])
        concentration_names = ['W', 'C1', 'C4']
        for i in range(len(pvalues)):
            if pvalues[i] <= 0.05:
                print('Significant Pair', concentration_names[pairs[i][0]], 'vs.', concentration_names[pairs[i][1]], 'pvalue =', pvalues[i], 't =', t_statistics[i])
            else:
                print('Non significant pair', concentration_names[pairs[i][0]], 'vs.', concentration_names[pairs[i][1]], 'pvalue =', pvalues[i], 't =', t_statistics[i])

        fig = plt.figure(figsize=(3, 4))
        plt.subplots_adjust(left=0.5)
        fig.add_axes([0.2, 0.2, .4, .4])

        if zscore_data:
            multiplicative_factor = 1
        else:
            multiplicative_factor = 100  # If not z-scoring, multiply everything by 100 to get percent dF/F

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

        plt.savefig('FigurePdfs/SupplementaryFigures/' + header + 'F_vs_ConcentrationSummary_5_to_10_window.pdf', transparent=True)
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

    # SUCROSE + QUININE Experiments

    'Carousel Experiment, Sucrose and Quinine, mPFC': {
        'Baseline Window': mPFC_baseline_window,
        'Example dataset': -1,
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
        baseline_window=dataset['Baseline Window'],
        averaging_window=[5, 10],
    )
# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Produces the non-heatmap plots of from the isosbestic fluorescence and tracking data of mice exploring the EPM

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import utils.prettyplot as prettyplot
from statsmodels.stats.multitest import multipletests as mult_test

def perform_analysis_for_experiment(experiment):

    datasets = experiment['data']
    condition = experiment['Experiment Name']

    dataset = datasets[0]
    center = dataset['EPM center']
    arm_length = dataset['EPM Arm length']
    center_radius = dataset['EPM Center radius']

    def zone(x, y, center, center_radius, arm_length):

        vertical_band =   np.logical_and(np.abs(x) <= center_radius, np.abs(y) <= arm_length)
        horizontal_band = np.logical_and(np.abs(y) <= center_radius, np.abs(x) <= arm_length)

        right_zone = np.logical_and(x >= center_radius, horizontal_band)
        left_zone = np.logical_and(x <= -center_radius, horizontal_band)
        top_zone = np.logical_and(y >= center_radius, vertical_band)
        bottom_zone = np.logical_and(y <= -center_radius, vertical_band)
        center_zone = np.logical_and(vertical_band, horizontal_band)

        return right_zone, left_zone, top_zone, bottom_zone, center_zone


    def get_zone_means(d, center, center_radius, arm_length):

        x = d['x_coordinate']
        y = d['y_coordinate']
        f = d['dF/F Isosbestic']

        closed_familiar_mask, closed_novel_mask, t, b, center_mask = zone(x, y, center, center_radius, arm_length)

        open_mask = np.logical_or(t, b)

        if np.sum(open_mask) > 0:
            open_mean = np.mean(f[open_mask])
        else:
            open_mean = np.NAN

        if np.sum(closed_novel_mask) > 0:
            novel_closed_mean = np.mean(f[closed_novel_mask])
        else:
            novel_closed_mean = np.NAN

        if np.sum(closed_familiar_mask) > 0:
            familiar_closed_mean = np.mean(f[closed_familiar_mask])
        else:
            familiar_closed_mean = np.NAN

        if np.sum(center_mask) > 0:
            center_mean = np.mean(f[center_mask])
        else:
            center_mean = np.NAN

        return familiar_closed_mean, novel_closed_mean, open_mean, center_mean

    familiar_closed_means = []
    novel_closed_means = []
    open_means = []
    center_means = []

    for d in datasets:
        familiar_closed_mean, novel_closed_mean, open_mean, center_mean = get_zone_means(d, np.array([0, 0]), center_radius, arm_length)
        familiar_closed_means.append(familiar_closed_mean)
        novel_closed_means.append(novel_closed_mean)
        open_means.append(open_mean)
        center_means.append(center_mean)

    familiar_closed_means = np.array(familiar_closed_means)
    novel_closed_means = np.array(novel_closed_means)
    open_means = np.array(open_means)
    center_means = np.array(center_means)


    def nansem(x):
        x = x[~np.isnan(x)]
        return np.std(x)/np.sqrt(len(x))

    fig = plt.figure(figsize=(1.5, 3))
    plt.subplots_adjust(left=0.24, bottom=0.2)
    plt.scatter(familiar_closed_means*0 + 0, 100*familiar_closed_means, color=[0.8, 0.8, 0.8])
    plt.scatter(novel_closed_means*0 + 1, 100*novel_closed_means, color=[0.8, 0.8, 0.8])
    plt.scatter(center_means*0 + 2, 100*center_means, color=[0.8, 0.8, 0.8])
    plt.scatter(open_means*0 + 3, 188*open_means, color=[0.8, 0.8, 0.8])
    mean_list = [100*np.nanmean(familiar_closed_means), 100*np.nanmean(novel_closed_means), 100*np.nanmean(center_means), 100*np.nanmean(open_means)]
    sem_list = [100*nansem(familiar_closed_means), 100*nansem(novel_closed_means), 100*nansem(center_means), 100*nansem(open_means)]
    plt.errorbar([0, 1, 2, 3], mean_list, yerr=sem_list, color='k', marker='o', capsize=5, linestyle='')
    prettyplot.no_box()
    prettyplot.ylabel('%dF/F')
    plt.xticks([0, 1, 2, 3], ['Clf', 'Cln', 'Ce', 'Op'])
    plt.ylim([-5, 20])

    plt.savefig('FigurePdfs/Isosbestic/' + condition + '_EPMSummary_Isosbestic.pdf', transparent=True)
    plt.show()


    stats = [
        scipy.stats.ttest_rel(familiar_closed_means, novel_closed_means),
        scipy.stats.ttest_rel(familiar_closed_means, center_means),
        scipy.stats.ttest_rel(familiar_closed_means, open_means)
    ]

    pvalues = [s.pvalue for s in stats]

    corrected_pvalues = mult_test(pvalues, method='bonferroni')[1]

    print('pvalues comparing different regions of the EPM with the familiar closed arm')
    print(condition + ': novel closed;, p =', corrected_pvalues[0], 'full stats (uncorrected) =', stats[0])
    print(condition + ': center; p =', corrected_pvalues[1], 'full stats (uncorrected) =', stats[1])
    print(condition + ': open; p =', corrected_pvalues[2], 'full stats (uncorrected) =', stats[2])

with open('../PreprocessedData/CompleteEPMDataset.obj', "rb") as input_file:
    EPM_Complete_Dataset = pickle.load(input_file)

perform_analysis_for_experiment(EPM_Complete_Dataset['EPM_mPFC'])
perform_analysis_for_experiment(EPM_Complete_Dataset['EPM_NAcCore'])



















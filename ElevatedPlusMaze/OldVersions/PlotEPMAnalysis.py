import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.collections as collections
import matplotlib.patches as patches
import scipy.stats
import utils.prettyplot as prettyplot
import os
import scipy.signal as signal
from statsmodels.stats.multitest import multipletests as mult_test
#from ElevatedPlusMaze.auxiliaryfunctions import EPMGeometry
import auxiliaryfunctions.transformtracking

def perform_analysis_for_condition(condition):

    dataset, average_width, center, fraction_of_radius_in_center, arm_length, center_radius = auxiliaryfunctions.transformtracking.load_dataset_and_rescale_tracking('FearConditioningData/' + condition + '/')

    def plot_EPM_Example(data, filename):
        downsample = 100
        d = data['F_ex']
        ts = data['ts']
        x = data['x_coordinate']
        y = data['y_coordinate']
        sos = signal.bessel(2, 50, fs=data['fs'], output='sos')
        d = signal.sosfiltfilt(sos, d)
        x = signal.sosfiltfilt(sos, x)
        y = signal.sosfiltfilt(sos, y)
        d = d*100
        d = d[::downsample]
        ts = ts[::downsample]
        x = x[::downsample]
        y = y[::downsample]



        # vmin = np.min(d)
        # vmax = np.max(d)
        # d = d - vmin
        # d = d/(vmax - vmin)

        # plt.hist(d, 100)
        # plt.show()

        d_sorted = np.sort(d)
        n = d_sorted.shape[0]
        vmin = d_sorted[n//64]
        vmax = d_sorted[n - n//64]
        d = d - vmin
        d = d/(vmax - vmin)

        # plt.hist(d, 100)
        # plt.show()

        plasma = matplotlib.colormaps['plasma']

        segments = []
        colors = []
        for i in range(len(d) - 1):
            segments.append([(x[i], y[i]), (x[i + 1], y[i + 1])])
            colors.append(plasma(0.5 * (d[i] + d[i + 1])))

        scalar_mappable = matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps['plasma'])
        scalar_mappable.set_clim(vmin, vmax)
        colorbar = plt.colorbar(scalar_mappable)
        lc = collections.LineCollection(segments, colors=colors, linewidths=1)
        plt.gca().add_collection(lc)
        plt.gca().autoscale()
        plt.gca().set_aspect(1)

        prettyplot.no_axes()
        prettyplot.title(filename)

    plt.figure()

    which_example_dataset = 0

    plot_EPM_Example(dataset[which_example_dataset], 'Dataset 1')
    rect1 = patches.Rectangle((center[0] - arm_length, center[1] - center_radius), arm_length * 2, center_radius * 2, color=[0.9, 0.9, 0.9])
    rect2 = patches.Rectangle((center[0] - center_radius, center[1] - arm_length), center_radius * 2, arm_length * 2, color=[0.9, 0.9, 0.9])
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.savefig('FigurePdfs/' + condition + '_EPMSingleAnimalExample.pdf', transparent=True)
    plt.show()





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
        f = d['F_ex']

        print(x.shape, y.shape, f.shape)

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

    for d in dataset:
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

    plt.savefig('FigurePdfs/' + condition + '_EPMSummary.pdf', transparent=True)
    plt.show()


    stats = [scipy.stats.ttest_rel(familiar_closed_means, novel_closed_means),
     scipy.stats.ttest_rel(familiar_closed_means, center_means),
     scipy.stats.ttest_rel(familiar_closed_means, open_means)]

    pvalues = [s.pvalue for s in stats]

    corrected_pvalues = mult_test(pvalues, method='bonferroni')[1]

    print('pvalues comparing different regions of the EPM with the familiar closed arm')
    print(condition + ': novel closed;, p =', corrected_pvalues[0], 'full stats (uncorrected) =', stats[0])
    print(condition + ': center; p =', corrected_pvalues[1], 'full stats (uncorrected) =', stats[1])
    print(condition + ': open; p =', corrected_pvalues[2], 'full stats (uncorrected) =', stats[2])

perform_analysis_for_condition('mPFC')
perform_analysis_for_condition('NAcCore')
perform_analysis_for_condition('mPFC_GFP')



















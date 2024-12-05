import matplotlib.pyplot as plt
import scipy.stats as stats
import utils.prettyplot as prettyplot
import utils.statsfunctions
from OriginalData.DrawerExperimentsData.InteractionTimes import *



# Plot the revisit interaction times

def plot_interaction_times(times_dict, ylabel, plot_order=None):

    plt.figure(figsize=(4, 2))
    plt.subplots_adjust(bottom=0.3)
    means = []
    sems = []
    widths = []
    positions = []
    ticks = []
    tick_labels = []
    current_position = 0
    large_width = 0.75  # bar width
    small_width = 0.33  # half-bar width
    colors = []
    if plot_order == None:
        keys = list(times_dict.keys())
    else:
        keys = plot_order
    for key in keys:
        if 'estrus' in key and not ('diestrus' in key):
            colors.append(np.array(prettyplot.colors['red']))
            widths.append(large_width)
            positions.append(current_position)
            ticks.append(current_position)
            tick_labels.append(key[:-8])
            current_position += 1

        elif 'diestrus' in key:
            colors.append(np.array(prettyplot.colors['blue']))
            positions.append(current_position)
            widths.append(large_width)
            ticks.append(current_position)
            tick_labels.append(key[:-10])
            current_position += 1
        elif key[0] == 'M':
            colors.append(np.array(prettyplot.colors['green']))
            positions.append(current_position)
            widths.append(large_width)
            ticks.append(current_position)
            tick_labels.append(key)
            current_position += 1

        else:
            colors.append(np.array(prettyplot.colors['black']))
            positions.append(current_position)
            widths.append(large_width)
            ticks.append(current_position)
            tick_labels.append(key)
            current_position += 1


        means.append(np.mean(times_dict[key]))
        sems.append(np.std(times_dict[key]) / np.sqrt(len(times_dict[key])))
    xvals = np.arange(len(times_dict.keys()))
    for i in range(len(positions)):
        plt.errorbar(positions[i], means[i], yerr=sems[i], capsize=5, linestyle='', color=colors[i])

    plt.bar(positions, means, width=widths, color=colors)
    prettyplot.ylabel(ylabel)

    pvals = []
    for i, key in enumerate(times_dict.keys()):
        if i > 0:
            pval = stats.ttest_ind(times_dict[key], times_dict['control switch']).pvalue
            pvals.append(pval)

    pvals = utils.statsfunctions.apply_bonferroni_correction(pvals)

    for i in range(len(times_dict) - 1):
        if pvals[i] < 0.01:
            plt.text(positions[i + 1], means[i + 1] + sems[i + 1] + 2, '**', fontsize=18, horizontalalignment='center', verticalalignment='center')
        elif pvals[i] < 0.05:
            plt.text(positions[i + 1], means[i + 1] + sems[i + 1] + 2, '*', fontsize=18, horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(positions[i + 1], means[i + 1] + sems[i + 1] + 2, 'N.S.', fontsize=12, horizontalalignment='center', verticalalignment='center')

    print('p values for', ylabel, list(pvals))

    plt.ylim([0, 50])
    plt.xticks(ticks, tick_labels, rotation=45)
    for tick in plt.gca().xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    prettyplot.no_box()
    prettyplot.ylabel(ylabel)

plot_order = [
    'control switch',
    'novel floor',
    'novel object',
    'M ECC',
    'M-M',
    'M-F',
    'F, ECC, diestrus',
    'F-F, diestrus',
    'F-M, diestrus',
    'F, ECC, estrus',
    'F-F, estrus',
    'F-M, estrus',

]



plot_interaction_times(interaction_times_dictionary_PFC, 'interaction time s', plot_order)
plt.savefig('FigurePdfs/mPFC_InteractionTimes.pdf', transparent=True)
plt.show()

plot_interaction_times(revisit_times_dictionary_PFC, 'revisit time s', plot_order)
plt.savefig('FigurePdfs/mPFC_RevisitTimes.pdf', transparent=True)
plt.show()
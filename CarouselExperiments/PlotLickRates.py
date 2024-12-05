import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import DataProcessingFunctions
import utils.statsfunctions as statsfunctions

Experiments = {
    'Quinine': {
        'data files': ['LickData/Carousel Experiment, Quinine, mPFC_mPFC_lick_counts.npy',
                       'LickData/Carousel Experiment, Quinine, NAc Core_NAc Core_lick_counts.npy']
    },
    'Quinine and sucrose': {
        'data files': ['LickData/Carousel Experiment, Sucrose and Quinine, mPFC_mPFC_lick_counts.npy',
                       'LickData/Carousel Experiment, Sucrose and Quinine, NAc Core_NAc_Core_lick_counts.npy']
    },
    'Sucrose': {
        'data files': ['LickData/Carousel Experiment, Sucrose, mPFC_mPFC_lick_counts.npy',
                       'LickData/Carousel Experiment, Sucrose, NAc Core_NAc_Core_lick_counts.npy']
    },
}

for key in Experiments.keys():
    if 'inin' in key:
        concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]
        solute = 'quinine'
    else:
        concentrations = [0, 29, 58, 117, 233]
        solute = 'sucrose'

    paths = Experiments[key]['data files']

    d = []
    for path in paths:
        d.append(np.load(path))

    y = np.concatenate(d, axis=1)

    fig = plt.figure(figsize=(2, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    plt.errorbar(concentrations, np.mean(y, 1), yerr=np.std(y, 1)/np.sqrt(y.shape[1]), capsize=5, marker='.', linestyle='', markersize=12, color='k')
    plt.gca().set_ylim(bottom=1)
    prettyplot.no_box()
    plt.xticks(concentrations)
    prettyplot.ylabel('number of licks')
    prettyplot.xlabel(solute + ' concentration mM')
    print(key)

    p, F = statsfunctions.repeated_measures_ANOVA(np.array([0, 1, 2, 3, 4]), y)
    print('Repeated measures ANOVA', 'n =', y.shape[1], 'F =', F, 'p =', p)

    pvalues, pairs, sign_ranks = DataProcessingFunctions.multi_wilcoxon_test_bonferroni_corrected(y[[0, 1, 4], :])

    Qs = ['W', 'C1', 'C4']
    for i in range(len(pvalues)):
        print('Pair', Qs[pairs[i][0]], 'vs.', Qs[pairs[i][1]], 'pvalue =', pvalues[i], 'sign_rank =', sign_ranks[i])


    plt.savefig('FigurePdfs/LickRates/' + key + '.pdf', transparent=True)
    plt.show()
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import DataProcessingFunctions

quinine_concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]      # concentrations of quinine in mM
PFC_LickCounts = np.load('PFC_lick_counts.npy')
NAcCore_LickCounts = np.load('NAcCore_lick_counts.npy')

y = np.concatenate([PFC_LickCounts, NAcCore_LickCounts], axis=1)

fig = plt.figure(figsize=(2, 4))
plt.subplots_adjust(left=0.25, bottom=0.25)

plt.errorbar(quinine_concentrations, np.mean(y, 1), np.std(y, 1)/np.sqrt(y.shape[1]), capsize=5, marker='.', linestyle='')
plt.ylim([1, 10])
prettyplot.no_box()
plt.xticks(quinine_concentrations)
prettyplot.ylabel('number of licks')
prettyplot.xlabel('quinine concentration mM')

pvalues, pairs = DataProcessingFunctions.multi_wilcoxon_test_bonferroni_corrected(y)

Qs = ['W', 'Q1', 'Q2', 'Q3', 'Q4']
for i in range(len(pvalues)):
    print('Pair', Qs[pairs[i][0]], 'vs.', Qs[pairs[i][1]], 'pvalue =', pvalues[i])


plt.savefig('FigurePdfs/AllAnimalLickAverages.pdf', transparent=True)
plt.show()
# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

import scipy.stats as stats
import numpy as np
from statsmodels.stats.multitest import multipletests as mult_test
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm

def linear_regression(x, y):
    """
    Compute the linear regression of y vs. x. Note: Assumes that the measurements in y are all independent. If they are
    from the same member of a group, use mixed_effect_linear_regression.
    :param x: A numpy array of shape [n]
    :param y: A numpy array of shape [n, r], where r = number of replicates
    :return: stats object from scipy's linregress function
    """

    x = np.array(x)
    y = np.array(y)
    x = x[:, None] * np.ones(shape=y.shape)
    return stats.linregress(np.reshape(x, [-1]), np.reshape(y, [-1]))

def ANOVA_one_way(y, equal_variances=True):
    """
    Compute an ANOVA for a collection of groups
    :param y: A numpy array of shape [n, r], where r = number of replicates
    :return: stats object from scipy's ANOVA
    """
    if equal_variances:
        if type(y) == list:
            anova_stats = stats.f_oneway(*y)
            return anova_stats.statistic, anova_stats.pvalue
        else:
            groups = []
            for i in range(y.shape[0]):
                groups.append(y[i, :])
            anova_stats = stats.f_oneway(*groups)
            return anova_stats.statistic, anova_stats.pvalue
    else:
        if type(y) == list:
            anova_stats = stats.alexandergovern(*y)
            return anova_stats.statistic, anova_stats.pvalue
        else:
            groups = []
            for i in range(y.shape[0]):
                groups.append(y[i, :])
            anova_stats = stats.alexandergovern(*groups)
            return anova_stats.statistic, anova_stats.pvalue

def repeated_measures_ANOVA(x, y):
    """
    Compute an ANOVA for the y as a function of x
    :param x: A numpy array of shape [n]
    :param y: A numpy array of shape [n, r], where r = number of repeated measures
    :return: p_value, F_value
    """

    data = {
        'x': np.repeat(x, y.shape[1]),
        'y': y.flatten(),
        'r': np.tile(np.arange(y.shape[1]), x.shape[0]) # separate the replicates into groups
    }

    stats_output = AnovaRM(data=pd.DataFrame(data), depvar='y', subject='r', within=['x']).fit()

    pvalue = stats_output.anova_table['Pr > F']['x']
    Fvalue = stats_output.anova_table['F Value']['x']

    return pvalue, Fvalue

def mixed_effect_linear_regression(x, y, individual_slopes=False):
    """
    Performs linear regression of y vs. x, where x has shape [n] and y has shape [n, r] where r is the number of
    replicates. Uses a mixed effect regression to take into account that the repeated measures are from the same animal.
    :param x: dependent variable
    :param y: measured effects
    :return: stats
    """
    data = {
        'x': np.repeat(x, y.shape[1]),
        'y': y.flatten(),
        'r': np.tile(np.arange(y.shape[1]), x.shape[0]) # separate the replicates into groups
    }

    data_frame = pd.DataFrame(data)
    if individual_slopes == False:
        model = mixedlm("y ~ x", data_frame, groups=data['r'])
    else:
        model = mixedlm("y ~ x", data_frame, groups=data['r'], re_formula="~x")
    result = model.fit()

    intercept = result.params['Intercept']
    slope = result.params['x']
    pvalue = result.pvalues['x']
    tvalue = result.tvalues['x']

    return {'intercept': intercept, 'slope': slope, 'pvalue': pvalue, 'tvalue': tvalue}

def display_n_sig_figs(x: float, n: int, smallest_non_scientific_number=0.001, largest_non_scientific_number=10000):
    """
    Make a string representing the number x keeping n significant figures.
    :param x: float
    :param n: Number of significant figures
    :param smallest_non_scientific_number: numbers smaller than this number are displayed in scientific notation.
    :param largest_non_scientific_number: numbers larger than this number are displayed in scientific notation.
    :return: str
    """
    if np.abs(x) >= smallest_non_scientific_number and np.abs(x) < largest_non_scientific_number:
        return np.format_float_positional(x, precision=n, unique=True, fractional=False, trim='k')
    else:
        return np.format_float_scientific(x, precision=n, unique=True)


def multi_pairwise_t_test_bonferroni_corrected(data):
    """
    Perform a pairwise t-test between every pair of the first index of data (the second index is replicates).
    Then correct the p-values with the Bonferroni correction

    :param data: a numpy array with axes (conditions, replicates)
    :return: the p-values and a list of the pairs (axis_i, axis_j)
    """
    pvals = []
    pairs = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            pairs.append((i, j))
            stats_data = stats.ttest_rel(data[i, :], data[j, :])
            pvals.append(stats_data.pvalue)
    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]
    return corrected_pvalues, pairs


def multi_t_test_bonferroni_corrected(data, equal_var=True):
    """
    Perform a pairwise t-test between every pair of the first index of data (the second index is replicates).
    Then correct the p-values with the Bonferroni correction

    :param data: a numpy array with axes (conditions, replicates)
    :return: the p-values and a list of the pairs (axis_i, axis_j)
    """
    pvals = []
    pairs = []
    number_of_groups = len(data)
    for i in range(number_of_groups - 1):
        for j in range(i + 1, number_of_groups):
            pairs.append((i, j))
            stats_data = stats.ttest_ind(data[i], data[j], equal_var=equal_var, nan_policy='omit')
            pvals.append(stats_data.pvalue)
    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]

    return corrected_pvalues, pairs

def apply_bonferroni_correction(pvals):
    """
    Applies the Bonferroni correction to a list of pvalues

    :param pvalues: a list or numpy array of pvalues
    :return: the corrected pvalues
    """

    corrected_pvalues = mult_test(pvals, method='bonferroni')[1]
    return corrected_pvalues
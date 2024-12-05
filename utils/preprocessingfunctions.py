# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
from utils.downsample import downsample
from utils.exponentialfit import fit_exponential_decay_to_data
import scipy.signal as signal
import warnings

def downsample_data_and_subtract_isosbestic(
        ts,
        fluorescence_excitation,
        fluorescence_isosbestic,
        artifact_removal_time=1.5,
        downsample_skip=10,
        plot_exponential_fits=False,
        dataset_name=''
):

    def remove_artifact(ts, f, t_0):
        I = np.logical_and(ts > t_0, ts < t_0 * 2)
        x = ts[I]
        y = f[I]
        c = np.polyfit(x, y, 1)
        indices_in_artifact_window = ts <= t_0
        f[indices_in_artifact_window] = ts[indices_in_artifact_window] * c[0] + c[1]

    f_ex = fluorescence_excitation * 1
    f_iso = fluorescence_isosbestic * 1
    t = ts * 1

    # Downsample the data
    if downsample_skip > 1:
        f_ex = downsample(f_ex, downsample_skip)
        f_iso = downsample(f_iso, downsample_skip)
        t = t[::downsample_skip]

    fs = 1/(t[1] - t[0])

    # Get rid of any spikes in the data that occur right after the photometry rig is turned on.
    # No analysis should occur during this window.
    if artifact_removal_time is not None:
        remove_artifact(t, f_ex, artifact_removal_time)
        remove_artifact(t, f_iso, artifact_removal_time)


    f_ex_0 = fit_exponential_decay_to_data(t, f_ex, 500)
    dF_over_F_f_ex = (f_ex - f_ex_0) / f_ex_0

    f_iso_0 = fit_exponential_decay_to_data(t, f_iso, 500)
    dF_over_F_f_iso = (f_iso - f_iso_0) / f_iso_0


    if plot_exponential_fits:
        fig, axes = plt.subplots(2, figsize=(3, 6))
        plt.subplots_adjust(left=0.2, bottom=0.2)

        axes[0].plot(t[::10], f_ex[::10], color='k', label='excitation')
        axes[0].plot(t[::10], f_ex_0[::10], color=prettyplot.colors['blue'], label='exponential fit')
        axes[0].legend(frameon=False)
        prettyplot.no_box(axes[0])
        prettyplot.title(dataset_name, axis=axes[0])
        prettyplot.ylabel('F', axis=axes[0])

        axes[1].plot(t[::10], f_iso[::10], color='k', label='isosbestic')
        axes[1].plot(t[::10], f_iso_0[::10], color=prettyplot.colors['blue'], label='exponential fit')
        axes[1].legend(frameon=False)
        prettyplot.no_box(axes[1])
        prettyplot.xlabel('time s', axis=axes[1])
        prettyplot.ylabel('F', axis=axes[1])

        plt.show()

    isosbestic_fit_coefficients = np.polyfit(dF_over_F_f_iso, dF_over_F_f_ex, 1)

    if isosbestic_fit_coefficients[0] < 0:
        warnings.warn('Negative isosbestic fit coefficient found. Setting to zero.')
        isosbestic_fit_coefficients = (0, isosbestic_fit_coefficients[1])



    f_iso_fit = dF_over_F_f_iso * isosbestic_fit_coefficients[0] + isosbestic_fit_coefficients[1]

    dF_over_F_with_subtraction = dF_over_F_f_ex - f_iso_fit

    explained_variance = 1 - np.std(dF_over_F_with_subtraction)/np.std(dF_over_F_f_ex)

    return_dictionary = {
        'times': t,
        'fs': fs,
        'F Excitation': f_ex,
        'F Isosbestic': f_iso,
        'dF/F Excitation': dF_over_F_f_ex,
        'dF/F Isosbestic': dF_over_F_f_iso,
        'dF/F Isosbestic fit to Excitation': f_iso_fit,
        'dF/F Excitation after isosbestic subtraction': dF_over_F_with_subtraction,
        'Explained variance (excitation signal explained by isosbestic)': explained_variance,
    }

    return return_dictionary
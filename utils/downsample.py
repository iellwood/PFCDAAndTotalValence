# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

import numpy as np
import scipy.signal as signal

def downsample(x, skip:int):
    """
    Decimates the signal after using a bessel filter to lowpass the data
    :param x: data
    :param skip: amount of decimation. skip = 10 corresponds to every 10th data point being kept.
    :return: downsampled data
    """
    x = np.array(x)
    downsample_filter = signal.bessel(8, [0.5/skip], output='sos', fs=1)
    x = signal.sosfiltfilt(downsample_filter, x)
    x = x[::skip]
    return x


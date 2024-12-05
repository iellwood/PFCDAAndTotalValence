import numpy as np
import scipy.signal as signal

def downsample(x, skip:int):
    x = np.array(x)
    downsample_filter = signal.bessel(8, [0.5/skip], output='sos', fs=1)
    x = signal.sosfiltfilt(downsample_filter, x)
    x = x[::skip]
    return x


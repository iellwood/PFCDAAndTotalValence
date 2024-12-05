# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

import numpy as np
from scipy.optimize import curve_fit
import torch
import matplotlib.pyplot as plt

def fit_exponential_decay_to_data(x, y, training_steps=500, print_fit_quality_during_training=False):
    '''
    Fits a double exponential c + A e^(-B x) + C e^(-D x).
    N.B. The coefficients are all constrained to be positive.

    As a fallback, a line is fitted to the data with the slope constrained to be negative.
    If the explained variance of the linear fit is higher than the exponential fit, the linear fit is
    used. Note that a descending line is a limit of the parameter space of the exponential fit.

    If the fitted line has a positive slope, the mean of y is used instead.

    :param x:
    :param y: Dependent variable
    :return: fit
    '''

    coeffs = np.polyfit(x, y, 1)

    if coeffs[0] > 0:  # Ensure that the slope is negative. Otherwise, just use the mean of y
        coeffs[0] = 0
        coeffs[1] = np.mean(y)

    y_linear_fit = x * coeffs[0] + coeffs[1]

    explained_variance_linear_fit = 1 - np.std(y - y_linear_fit)/np.std(y)


    p_0 = [
        np.mean(y[-y.shape[0]//2:]),
        .25 * (np.mean(y[:y.shape[0]//2]) - np.mean(y[y.shape[0]//2:])),
        1/(x[-1] - x[0]),
        0.75 * (np.mean(y[:y.shape[0]//2]) - np.mean(y[y.shape[0]//2:])),
        10/(x[-1] - x[0])
    ]

    p_0 = np.maximum(p_0, 0)

    p_0 = torch.tensor(p_0, dtype=torch.float, requires_grad=False)
    x = torch.tensor(x, requires_grad=False, dtype=torch.float)
    y = torch.tensor(y.copy(), requires_grad=False, dtype=torch.float)

    def f(x):
        return torch.where(x > 0, x, 1/(1 - x))

    def exp_with_constant(x, p):
        return p[0] + p[1] * torch.exp(-p[2] * x) + p[3] * torch.exp(-p[4] * x)

    p = torch.tensor(p_0.detach().numpy() * 0 + 0, dtype=torch.float, requires_grad=True)

    y_mean = torch.mean(y)
    y_std = torch.std(y)

    optimizer = torch.optim.Adam(params=[p], lr=0.1)

    for i in range(training_steps + 1):
        optimizer.zero_grad()
        y_model = exp_with_constant(x, p_0 * f(p))
        loss = torch.mean(torch.square(y_model - y + torch.normal(0, y_std/((i + 1)), size=y.shape) ))
        loss.backward()
        optimizer.step()



        if print_fit_quality_during_training: # Set to true to print out the explained variance every 100 training steps
            if i % 100 == 0:
                explained_var = 1 - torch.std(y_model - y)/torch.std(y)
                #print('Fitting Exponential, step =', i, 'explained var =', np.round(explained_var.detach().numpy() * 100, 1))


    y_model = exp_with_constant(x, p_0 * f(p))
    explained_var_exponential_fit = 1 - torch.std(y_model - y) / torch.std(y)

    if explained_var_exponential_fit.detach().cpu().numpy() > explained_variance_linear_fit:

        return y_model.detach().cpu().numpy()

    else: # fall back to a simple linear fit if the exponential fit was worse.

        return y_linear_fit



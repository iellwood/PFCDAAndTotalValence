# This file preprocesses the signals recorded during the Elevated Plus Maze experiment.
#
# Notes:
# 1) Units for the fluorescence are in mV from the Newport femtowatt detector on the photometry rig.
# 2) The data is low-passed filtered and decimated by a factor of 10 to save space and speed up the analysis. (This
# makes the sampling frequency a bit above 100 Hz.)
# 3) FearConditioningData from 0 to 1.5 seconds has been replaced by a linear interpolation to remove artifacts when the
# photometry rig is first turned on.No Analysis should include the data before t = 1.5 seconds.
# 4) dF/F for both the excitation signal and isosbestic was computed by fitting a double exponential to the signal
# and treating this as F_0.dF/F = (F - F_0) /F_0.
# 5) The dF/F of the isosbestic was linearly fit to dF/F of the excitation using np.polyfit.The resulting fit
# was subtracted from excitation.The fit is stored in 'dF/F Isosbestic fit to Excitation'.
# 6) The importance of the isosbestic subtraction is measured by 'Explained variance from isosbestic'.

import pickle
import numpy as np
import os
from auxiliaryfunctions.transformtracking import load_dataset_and_rescale_tracking
import matplotlib.pyplot as plt
from utils.preprocessingfunctions import downsample_data_and_subtract_isosbestic

downsample_skip = 10

Experiments = {
    'EPM_mPFC': {
        'Experiment Name': 'EPM_mPFC',
        'Experiment Detailed Name': 'Elevated Plus Maze (EPM), mPFC',
        'data path': 'FearConditioningData/mPFC/',
        'Brain Region': 'mPFC',
    },

    'EPM_NAcCore': {
        'Experiment Name': 'EPM_NAcCore',
        'Experiment Detailed Name': 'Elevated Plus Maze (EPM), NAc Core',
        'Brain Region': 'NAc Core',
        'data path': 'FearConditioningData/NAcCore/',
     },

    'EPM_mPFC_GFP': {
        'Experiment Name': 'EPM_GFP_Control',
        'Experiment Detailed Name': 'Elevated Plus Maze (EPM), mPFC, GFP Control',
        'data path': 'FearConditioningData/mPFC_GFP/',
        'Brain Region': 'mPFC',
    },
}

for key in Experiments.keys():

    experiment = Experiments[key]
    print('Preprocessing --', key)

    data_dictionaries = []

    dataset, average_width, center, fraction_of_radius_in_center, arm_length, center_radius = load_dataset_and_rescale_tracking(experiment['data path'])

    for i, d in enumerate(dataset):
        f_ex = d['F_ex']
        f_iso = d['F_iso']

        fs = d['fs']

        ts = np.arange(f_ex.shape[0]) / fs

        if f_ex.shape[0] != d['x_coordinate'].shape[0]:
            print('Tracking and photometry are not the same size')
            exit()
        data_dict = downsample_data_and_subtract_isosbestic(
            ts=ts,
            fluorescence_excitation=f_ex,
            fluorescence_isosbestic=f_iso,
            downsample_skip=downsample_skip,
            artifact_removal_time=None,
            plot_exponential_fits=False,
            dataset_name=key + '_animal_' + str(i)
        )



        data_dict['Readme'] = \
            """
            1) Units for the fluorescence are in mV from the Newport femtowatt detector on the photometry rig.
            2) The data is low-passed filtered and decimated by a factor of 10 to save space and speed up the analysis. (This
            makes the sampling frequency a bit above 100 Hz.)
            3) dF/F for both the excitation signal and isosbestic was computed by fitting a double exponential to the signal
            and treating this as F_0. dF/F = (F - F_0)/F_0.
            4) The dF/F of the isosbestic was linearly fit to dF/F of the excitation using np.polyfit. The resulting fit
            was subtracted from excitation. The fit is stored in 'dF/F Isosbestic fit to Excitation'.
            5) The importance of the isosbestic subtraction is measured by 'Explained variance from isosbestic'.
            """

        data_dict['x_coordinate'] = d['x_coordinate'][::downsample_skip]
        data_dict['y_coordinate'] = d['y_coordinate'][::downsample_skip]
        data_dict['Mouse ID'] = d['Mouse_ID']
        data_dict['EPM center'] = center
        data_dict['EPM Arm length'] = arm_length
        data_dict['EPM Arm width'] = average_width
        data_dict['EPM Fraction of radius in center'] = fraction_of_radius_in_center
        data_dict['EPM Center radius'] = center_radius

        data_dictionaries.append(data_dict)

    experiment['data'] = data_dictionaries

# Save the complete preprocessed dataset
with open('../PreprocessedData/CompleteEPMDataset.obj', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(Experiments, file)
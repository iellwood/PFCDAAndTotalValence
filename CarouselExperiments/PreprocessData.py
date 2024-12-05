# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# **WARNING**: This script requires the original, unprocessed data files, which must be in the folder OriginalData
# These files are not included in the GitHub repository due to their size, but can be made available upon request.

# This file preprocesses the signals recorded during the carousel experiment.
#
# Notes:
# 1) Units for the fluorescence are in mV from the Newport femtowatt detector on the photometry rig.
# 2) The data is low-passed filtered and decimated by a factor of 10 to save space and speed up the analysis. (This
# makes the sampling frequency a bit above 100 Hz.)
# 3) Data from 0 to 1.5 seconds has been replaced by a linear interpolation to remove artifacts when the
# photometry rig is first turned on.No Analysis should include the data before t = 1.5 seconds.
# 4) dF/F for both the excitation signal and isosbestic was computed by fitting a double exponential to the signal
# and treating this as F_0.dF/F = (F - F_0) /F_0.
# 5) The dF/F of the isosbestic was linearly fit to dF/F of the excitation using np.polyfit.The resulting fit
# was subtracted from excitation.The fit is stored in 'dF/F isosbestic fit to Excitation'.
# 6) The importance of the isosbestic subtraction is measured by 'Explained variance from isosbestic'.

import pickle
import numpy as np
import os
from DataProcessingFunctions import map_port_numbers_to_dose

from utils.preprocessingfunctions import downsample_data_and_subtract_isosbestic

quinine_concentrations = [0.0, 0.8, 1.6, 3.2, 4.8]
sucrose_concentrations = [0.0, 29.0, 58.0, 117.0, 233.0]      # concentrations of quinine in mM

Experiments = {
    'Carousel Experiment, Quinine, mPFC': {
        'Experiment Name': 'Carousel Experiment, Quinine, mPFC',
        'data path': 'FearConditioningData/Quinine/mPFC/',
        'Brain Region': 'mPFC',
        'Fixed Solute': 'none',
        'Fixed Solute concentration': 0,
        'Solute': 'quinine',
        'Solute Concentrations': quinine_concentrations,
    },

    'Carousel Experiment, Quinine, NAc Core': {
        'Experiment Name': 'Carousel Experiment, Quinine, NAc Core',
        'Brain Region': 'NAc Core',
        'data path': 'FearConditioningData/Quinine/NAcCore/',
        'Fixed Solute': 'none',
        'Fixed Solute concentration': 0,
        'Solute': 'quinine',
        'Solute Concentrations': quinine_concentrations,
     },

    # SUCROSE EXPERIMENTS

    'Carousel Experiment, Sucrose, mPFC': {
        'Experiment Name': 'Carousel Experiment, Sucrose, mPFC',
        'Brain Region': 'mPFC',
        'data path': 'FearConditioningData/Sucrose/mPFC/',
        'Fixed Solute': 'none',
        'Fixed Solute concentration': 0,
        'Solute': 'sucrose',
        'Solute Concentrations': sucrose_concentrations,
     },

    'Carousel Experiment, Sucrose, NAc Core': {
        'Experiment Name': 'Carousel Experiment, Sucrose, NAc Core',
        'Brain Region': 'NAc_Core',
        'data path': 'FearConditioningData/Sucrose/NAcCore/',
        'Fixed Solute': 'none',
        'Fixed Solute concentration': 0,
        'Solute': 'sucrose',
        'Solute Concentrations': sucrose_concentrations,
     },

    # SUCROSE + QUININE Experiments

    'Carousel Experiment, Sucrose and Quinine, mPFC': {
        'Experiment Name': 'Carousel Experiment, Sucrose and Quinine, mPFC',
        'Brain Region': 'mPFC',
        'data path': 'FearConditioningData/SucroseAndQuinine/mPFC/',
        'Fixed Solute': 'sucrose',
        'Fixed Solute concentration': -1,
        'Solute': 'quinine',
        'Solute Concentrations': quinine_concentrations,
     },

    'Carousel Experiment, Sucrose and Quinine, NAc Core': {
        'Experiment Name': 'Carousel Experiment, Sucrose and Quinine, NAc Core',
        'Brain Region': 'NAc_Core',
        'data path': 'FearConditioningData/SucroseAndQuinine/NAcCore/',
        'Fixed Solute': 'sucrose',
        'Fixed Solute concentration': -1,
        'Solute': 'quinine',
        'Solute Concentrations': quinine_concentrations,
     },
}

for key in Experiments.keys():
    experiment = Experiments[key]
    print('Preprocessing --', key)

    # Load the data
    file_names = os.listdir(experiment['data path'])
    data = []
    for file_name in file_names:
        with open(experiment['data path'] + file_name, "rb") as input_file:
            data.append(pickle.load(input_file))

    data_dictionaries = []

    for i, d in enumerate(data):
        f_ex = d['F_ex_original']
        f_iso = d['F_iso_original']
        fs = d['fs']
        ts = np.arange(f_ex.shape[0]) / fs

        data_dict = downsample_data_and_subtract_isosbestic(
            ts=ts,
            fluorescence_excitation=f_ex,
            fluorescence_isosbestic=f_iso,
            artifact_removal_time=1.5,
            plot_exponential_fits=False,
            dataset_name=key + ' dataset #' + str(i)
        )
        data_dict['Readme'] = \
            """
            1) Units for the fluorescence are in mV from the Newport femtowatt detector on the photometry rig.
            2) The data is low-passed filtered and decimated by a factor of 10 to save space and speed up the analysis. (This 
            makes the sampling frequency a bit above 100 Hz.)
            3) FearConditioningData from 0 to 1.5 seconds has been replaced by a linear interpolation to remove artifacts when the 
            photometry rig is first turned on. No Analysis should include the data before t = 1.5 seconds.
            4) dF/F for both the excitation signal and isosbestic was computed by fitting a double exponential to the signal
            and treating this as F_0. dF/F = (F - F_0)/F_0.
            5) The dF/F of the isosbestic was linearly fit to dF/F of the excitation using np.polyfit. The resulting fit
            was subtracted from excitation. The fit is stored in 'dF/F Isosbestic fit to Excitation'.
            6) The importance of the isosbestic subtraction is measured by 'Explained variance from isosbestic'.
            """
        data_dict['Mouse ID'] = d['Mouse_ID']
        data_dict['port'] = map_port_numbers_to_dose(d['port'])
        data_dict['first_lick_time'] = d['first_lick_time']
        data_dict['water_time'] = d['water_time']
        data_dict['number of licks'] = d['lick_rate']

        data_dictionaries.append(data_dict)

    experiment['data'] = data_dictionaries

# Save the complete preprocessed dataset
with open('../PreprocessedData/CompleteCarouselDataset.obj', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(Experiments, file)
# This file preprocesses the signals recorded during the Drawer Experiment.
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

from utils.preprocessingfunctions import downsample_data_and_subtract_isosbestic


Experiments = {

    'Drawer_Control_Swap': {
        'Experiment Name': 'Drawer Experiment, Control Swap',
        'data path': 'FearConditioningData/ControlSwap/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_Novel_Floor': {
        'Experiment Name': 'Drawer Experiment, Mouse interacts with novel floor',
        'data path': 'FearConditioningData/NovelFloor/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_Novel_Object': {
        'Experiment Name': 'Drawer Experiment, Mouse interacts with novel object',
        'data path': 'FearConditioningData/NovelObject/mPFC/',
        'Brain Region': 'mPFC',
    },


    'Drawer_M_EC': {
        'Experiment Name': 'Drawer Experiment, Male interacts with empty cage',
        'data path': 'FearConditioningData/Male-EmptyCage/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_M_F': {
        'Experiment Name': 'Drawer Experiment, Male interacts with female mouse in cage',
        'data path': 'FearConditioningData/Male-Female/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_M_M': {
        'Experiment Name': 'Drawer Experiment, Male interacts with male mouse in cage',
        'data path': 'FearConditioningData/Male-Male/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_F_F_Diestrus': {
        'Experiment Name': 'Drawer Experiment, Female in Diestrus interacts with female mouse in cage',
        'data path': 'FearConditioningData/Female-Female-Diestrus/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_F_F_Estrus': {
        'Experiment Name': 'Drawer Experiment, Female in Estrus interacts with female mouse in cage',
        'data path': 'FearConditioningData/Female-Female-Estrus/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_F_M_Diestrus': {
        'Experiment Name': 'Drawer Experiment, Female in Diestrus interacts with male mouse in cage',
        'data path': 'FearConditioningData/Female-Male-Diestrus/mPFC/',
        'Brain Region': 'mPFC',
    },

    'Drawer_F_M_Estrus': {
        'Experiment Name': 'Drawer Experiment, Female in Estrus interacts with male mouse in cage',
        'data path': 'FearConditioningData/Female-Male-Estrus/mPFC/',
        'Brain Region': 'mPFC',
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
            downsample_skip=10,
            plot_exponential_fits=False,
            dataset_name=key + ' dataset #' + str(i)
        )

        data_dict['Readme'] = \
            """
            1) Units for the fluorescence are in mV from the Newport femtowatt detector on the photometry rig.
            2) The data is low-passed filtered and decimated by a factor of 10 to save space and speed up the analysis. (This 
            makes the sampling frequency a bit above 100 Hz.)
            3) FearConditioningData from 0 to 1.5 seconds has been replaced by a linear interpolation to remove artifacts when the 
            photometry rig is first turned on. ** No analysis should include the data before t = 1.5 seconds. **
            4) dF/F for both the excitation signal and isosbestic was computed by fitting a double exponential to the signal
            and treating this as F_0. dF/F = (F - F_0)/F_0.
            5) The dF/F of the isosbestic was linearly fit to dF/F of the excitation using np.polyfit. The resulting fit
            was subtracted from excitation. The fit is stored in 'dF/F Isosbestic fit to Excitation'.
            6) The importance of the isosbestic subtraction is measured by 'Explained variance from isosbestic'.
            """

        # For all the keys that are not related to the fluorescence recordings, copy them into the data_dict
        key_list = list(d.keys())
        key_list.remove('ts')
        key_list.remove('fs')
        key_list.remove('F_ex')
        key_list.remove('F_iso')
        key_list.remove('F_ex_original')
        key_list.remove('F_iso_original')
        for key_list_key in key_list:
            if key_list_key == 'red_light':
                data_dict['red light'] = d[key_list_key]
            elif key_list_key == 'event_name':
                pass
            elif key_list_key == 'Mouse_ID':
                data_dict['Mouse ID'] = d[key_list_key]
            elif key_list_key == 'obj_enter':
                data_dict['stimulus inserted'] = d[key_list_key]
            elif key_list_key == 'novel1':
                data_dict['drawer entered after stimulus insertion'] = d[key_list_key]
            elif key_list_key == 'novel2':
                data_dict['drawer entered after second stimulus insertion'] = d[key_list_key]
            elif key_list_key == 'empty1':
                data_dict['drawer revisited after stimulus removal'] = d[key_list_key]

            elif key_list_key == 'cage_empty_insert':
                data_dict['stimulus inserted (empty cage)'] = d[key_list_key]
            elif key_list_key == 'enter_empty':
                data_dict['drawer entered after stimulus insertion (empty cage)'] = d[key_list_key]
            elif key_list_key == 'empty2':
                data_dict['drawer revisited after stimulus removal (empty cage)'] = d[key_list_key]

            elif key_list_key == 'cage_male_insert':
                data_dict['stimulus inserted'] = d[key_list_key]
            elif key_list_key == 'enter_male':
                data_dict['drawer entered after stimulus insertion'] = d[key_list_key]
            elif key_list_key == 'empty1':
                data_dict['drawer revisited after stimulus removal'] = d[key_list_key]
            else:
                print('key_list_key:', key_list_key)
                data_dict[key_list_key] = d[key_list_key]

        data_dictionaries.append(data_dict)

    experiment['data'] = data_dictionaries

# Save the complete preprocessed dataset
with open('../PreprocessedData/CompleteDrawerExperimentDataset.obj', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(Experiments, file)
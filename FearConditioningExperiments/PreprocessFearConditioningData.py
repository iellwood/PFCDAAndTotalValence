# This file preprocesses the signals recorded during the fear conditioning experiment.
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

    # mPFC Experiments

    'FC_mPFC_Day_1': {
        'Experiment Name': 'Fear Conditioning, Day 1 (Conditioning and Extinction), mPFC',
        'data path': 'FearConditioningData/Day1/mPFC/',
        'Brain Region': 'mPFC',
    },

    'FC_mPFC_Day_2': {
        'Experiment Name': 'Fear Conditioning, Day 2 (Fear Extinction Memory), mPFC',
        'data path': 'FearConditioningData/Day2/mPFC/',
        'Brain Region': 'mPFC',
    },

    # NAc Core Experiments

    'FC_NAcCore_Day_1': {
        'Experiment Name': 'Fear Conditioning, Day 1 (Conditioning and Extinction), NAc Core',
        'data path': 'FearConditioningData/Day1/NAcCore/',
        'Brain Region': 'NacCore',
    },

    'FC_NAcCore_Day_2': {
        'Experiment Name': 'Fear Conditioning, Day 2 (Fear Extinction Memory), NAc Core',
        'data path': 'FearConditioningData/Day2/NAcCore/',
        'Brain Region': 'NAcCore',
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

        # Exclude one animal from Day 1 due to a giant fluorescence deviation midway through the trial.
        # This deviation occurs in both the isosbestic and excitation signals and thus could, in principle
        # be included in our analysis, however we have removed it out of caution.
        if not (key == 'FC_mPFC_Day_1' and i == 6):

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
                photometry rig is first turned on. No Analysis should include the data before t = 1.5 seconds.
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
                if not 'tone' in key_list_key:
                    data_dict[key_list_key] = d[key_list_key]
                else:
                    if key_list_key == 'tone_test':
                        data_dict['CS (Unpaired)'] = d['tone_test']
                    elif key_list_key == 'tone1':
                        if 'Day_2' in key and 'NAc' in key:  # This fixes an inconsistency in the labelling of the unpaired CS between mPFC and NAcCore
                            data_dict['CS (Unpaired)'] = d['tone1']
                        else:
                            data_dict['CS (Paired)'] = d['tone1']
                    elif key_list_key == 'tone2':
                        data_dict['NCS'] = d['tone2']
                    else:
                        print('unhandled case')
                        exit()


            data_dictionaries.append(data_dict)

    experiment['data'] = data_dictionaries

# Save the complete preprocessed dataset
with open('../PreprocessedData/CompleteFearConditioningDataset.obj', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(Experiments, file)
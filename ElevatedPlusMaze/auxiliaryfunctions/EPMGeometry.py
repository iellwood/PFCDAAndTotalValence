# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Contains the pixel locations of various parts of the EPM. Used to translate and scale the EPM tracking data so that it
# is consistent across all experiments.

coordinates ={
    'mPFC_1': {
        'center': [850, 511],
        'left': [500, 511],
        'right': [1200, 511],
        'up': [850, 160],
        'down': [850, 860],
        'width': 65,
    },
    'mPFC_2': {
        'center': [570, 360],
        'left': [265, 360],
        'right': [875, 360],
        'up': [570, 60],
        'down': [570, 675],
        'width': 65,
    },
    'mPFC_3': {
        'center': [490, 365],
        'left': [60, 365],
        'right': [820, 365],
        'up': [490, 30],
        'down': [490, 710],
        'width': 65,
    },
    'NAcCore_1': {
        'center': [1125, 535],
        'left': [710, 535],
        'right': [1580, 535],
        'up': [1125, 120],
        'down': [1125, 975],
        'width': 80,
    },
    'NAcCore_2': {
        'center': [860, 530],
        'left': [420, 530],
        'right': [1270, 530],
        'up': [860, 115],
        'down': [860, 950],
        'width': 80,
    },
    'NAcCore_3': {
        'center': [880, 560],
        'left': [450, 560],
        'right': [1300, 560],
        'up': [880, 140],
        'down': [880, 985],
        'width': 80,
    },
    'NAcCore_4': {
        'center': [780, 530],
        'left': [340, 530],
        'right': [1200, 530],
        'up': [780, 110],
        'down': [780, 950],
        'width': 80,
    },
    'NAcCore_5': {
        'center': [850, 525],
        'left': [420, 525],
        'right': [1260, 525],
        'up': [850, 110],
        'down': [850, 950],
        'width': 80,
    },
}

# This dictionary gives the location of the EPM in the camera's view.
EPM_coordinate_dict = {

    # mPFC Animals
    'MX1_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX3_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX5_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX6_EPM_cleaned.obj': coordinates['mPFC_3'],
    'MX7_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX8_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX14_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX16_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX17_EPM_cleaned.obj': coordinates['mPFC_1'],
    'MX51_EPM_cleaned.obj': coordinates['mPFC_3'],
    'MX54_EPM_cleaned.obj': coordinates['mPFC_2'],
    'MX55_EPM_cleaned.obj': coordinates['mPFC_2'],
    'MX71_EPM_cleaned.obj': coordinates['mPFC_3'],
    'MX81_EPM_cleaned.obj': coordinates['mPFC_3'],

    #'MX66_EPM_cleaned.obj': coordinates['mPFC_2'],

    # NAcCore Animals
    'C1_EPM_cleaned.obj': coordinates['NAcCore_1'],
    'C4_EPM_cleaned.obj': coordinates['NAcCore_1'],
    'FC2_EPM_cleaned.obj': coordinates['NAcCore_3'],
    'FC3_EPM_cleaned.obj': coordinates['NAcCore_4'], # FearConditioningData has large baseline drift
    'FC4_EPM_cleaned.obj': coordinates['NAcCore_3'],
    'FC7_EPM_cleaned.obj': coordinates['NAcCore_3'],
    'FC8_EPM_cleaned.obj': coordinates['NAcCore_1'],
    'FC9_EPM_cleaned.obj': coordinates['NAcCore_2'],
    'FC10_EPM_cleaned.obj': coordinates['NAcCore_3'],
    'FC11_EPM_cleaned.obj': coordinates['NAcCore_5'],
    'FC12_EPM_cleaned.obj': coordinates['NAcCore_3'],

    'GFP2_EPM_cleaned.obj': coordinates['mPFC_1'],
    'GFP_EPM_cleaned.obj': coordinates['mPFC_2'],
}

EPM_start_side_dict = { # Checked by hand from the videos

    # mPFC Animals
    'MX1_EPM_cleaned.obj': 'right',
    'MX3_EPM_cleaned.obj': 'right',
    'MX5_EPM_cleaned.obj': 'right',
    'MX6_EPM_cleaned.obj': 'left',
    'MX7_EPM_cleaned.obj': 'right',
    'MX8_EPM_cleaned.obj': 'right',
    'MX14_EPM_cleaned.obj': 'left',
    'MX16_EPM_cleaned.obj': 'left',
    'MX17_EPM_cleaned.obj': 'left',
    'MX51_EPM_cleaned.obj': 'left',
    'MX54_EPM_cleaned.obj': 'right',
    'MX55_EPM_cleaned.obj': 'right',
    'MX71_EPM_cleaned.obj': 'left',
    'MX81_EPM_cleaned.obj': 'left',

    # NAcCore Animals
    'C1_EPM_cleaned.obj': 'down',
    'C4_EPM_cleaned.obj': 'down',
    'FC2_EPM_cleaned.obj': 'up',
    'FC3_EPM_cleaned.obj': 'down',
    'FC4_EPM_cleaned.obj': 'up',
    'FC7_EPM_cleaned.obj': 'up',
    'FC8_EPM_cleaned.obj': 'down',
    'FC9_EPM_cleaned.obj': 'down',
    'FC10_EPM_cleaned.obj': 'up',
    'FC11_EPM_cleaned.obj': 'down',
    'FC12_EPM_cleaned.obj': 'up',

    'GFP2_EPM_cleaned.obj': 'right',
    'GFP_EPM_cleaned.obj': 'right',
}

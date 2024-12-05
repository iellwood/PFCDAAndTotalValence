# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Lists and processes the interaction times in the drawer chamber experiments. Note the use of Grubb's test to remove
# outliers as explained below.
#
# Note that female interactions with the cage by itself were measured twice, once in the female meets male experiments
# as a control and once in the female meets female experiments as a control. Naively these should be equivelent experiments
# and we have averaged the results of them together on a mouse by mouse basis, taking into account that some mice were
# only run in one of the two experiments. Both F-M and F-F experiments showed higher interaction times with the cage
# in estrus than in diestrus.
#
# Set the flag combine_combine_female_cage_interactions = False to keep the datasets separate.
#
# We found two outliers in our interaction times, 121 s and 323 s, which were identified by a Grubb test for outliers
# with alpha = 0.01. The same outliers are present for alpha = 0.05. Note that for the outlier detection, all
# interaction times were merged into a collection. This was done to avoid issues with outlier identification for small
# n.
#
# These were removed from the dataset, which
# reduced interaction times in the novel floor revisited times and novel object interaction times. This removal only
# affects that statistics for the novel object interaction time. Without the removal, the mean is larger, but no longer
# significantly different from the control condition due to the enormous variance of the dataset.
#
# The outlier removal procedure can be turned off by setting remove_outliers = False. The outlier_alpha can also be
# changed by setting outlier_alpha = 0.05, for example
#
# ECC = empty cage control

import numpy as np
combine_combine_female_cage_interactions = True
remove_outliers = True
outlier_alpha = 0.01

interaction_times_dictionary_PFC = {
    'control switch': [3, 4, 3, 2, 5, 3, 6, 5, 3, 5, 3, 4,],
    'novel floor': [20, 34, 11, 46, 50, 65, 10, 34, 44, 27, 72, 53,],
    'novel object': [11, 35, 15, 29, 18, 10, 45, 323, 26, 10, 46,],
    'M ECC': [10, 11, 21, 13, 53,],
    'M-M': [20, 28, 34, 36, 43, 7, 77, 8, 6, 5, 7,],
    'M-F': [12, 31, 12, 18, 19, 27, 35, 28, 49, 19,],
    'F-M, ECC, estrus': [41, 77, 54, 39, 40, 43, 24, 36, 25, 22, 24, ],
    'F-M, ECC, diestrus': [16, 20, 29, 25, 28, 27, 25, 13, 16, 17, 36, 17,],
    'F-M, estrus': [50, 23, 68, 20, 45, 51, 56, 67, 27, 26, 30,],
    'F-M, diestrus': [38, 18, 36, 16, 26, 18, 19, 33, 24, 21, 27, 20,],
    'F-F, ECC, estrus': [35, 37, 47, 17, 28, 24, 25, 23, 27],
    'F-F, ECC, diestrus': [6, 26, 15, 23, 35, 28, 7, 18, 53,],
    'F-F, estrus': [29, 27, 23, 10, 38, 73, 56, 43, 13,],
    'F-F, diestrus': [20, 20, 23, 18, 40, 19,  29, 28, 40,],
}

animal_list_for_empty_cage_experiments = {
    'F-M, ECC, estrus': ['PFC2', 'PFC5', 'PFC6', 'PFC7', 'PFC8', 'PFC9', 'PFC10', 'PFC11', 'PFC21', 'PFC23', 'PFC24',],
    'F-F, ECC, estrus': ['PFC2', 'PFC5', 'PFC6',         'PFC8', 'PFC9', 'PFC10', 'PFC11',                   'PFC24',],
    'F-M, ECC, diestrus': ['PFC1', 'PFC2', 'PFC3', 'PFC5', 'PFC6', 'PFC7', 'PFC8', 'PFC9', 'PFC10', 'PFC11', 'PFC21', 'PFC24', ],
    'F-F, ECC, diestrus': [        'PFC2',         'PFC5', 'PFC6', 'PFC7', 'PFC8', 'PFC9', 'PFC10', 'PFC11',          'PFC24',],
}

control_interaction_times_dictionary_PFC = {
    'control switch': [],
    'novel floor': [12, 8, 3, 7, 9, 6, 5, 10, 13, 11, 18, 8,],
    'novel object': [17, 11, 2, 9, 8, 7, 48, 10, 38, 12,],
    'male meets male': [28, 12, 29, 9, 5, 4, 6, 38, 5, 20, 29,],
    'male meets female': [2, 4, 1, 2, 9, 5, 21, 9, 12, 6,],
    'female meets male, estrus': [5, 17, 10, 9, 7, 15, 8, 12, 9, 5, 13,],
    'female meets male, diestrus': [4, 12, 4, 3, 7, 30, 10, 7, 6, 3, 5, 3,],
    'female meets female, estrus': [7, 7, 6, 12, 7, 10, 8, 5, 7, 7,],
    'female meets female, diestrus': [5, 4, 19, 3, 12, 8, 8, 17, 9, 22,],
}

# This is the amount of time that the mouse spends in the drawer when it revisits the drawer after the stimulus
# has been withdrawn. Note that the in the drawer is cutoff if the animal begins grooming (indicating that it is no
# longer exploring.
revisit_times_dictionary_PFC = {
    'control switch': [4, 3, 3, 5, 10, 7, 4, 2, 8, 10, 14, 10,],
    'novel floor': [29, 40, 121, 18, 21, 19, 15, 15, 35, 57, 20, 32],
    'novel object': [17, 24, 18, 13, 38, 49, 52, 49, 14, 20, 74,],
    'M ECC': [15, 20, 14, 18, 40,],
    'M-M': [18, 22, 30, 18, 59, 19, 56, 22, 35, 57, 63,],
    'M-F': [9, 8, 42, 6, 27, 18, 50, 10, 14, 15,],
    'F-M, ECC, estrus': [36, 53, 43, 26, 27, 32, 18, 25, 18, 20, 21],
    'F-M, ECC, diestrus': [20, 17, 26, 20, 24, 19, 20, 22, 18, 22, 33, 18],
    'F-M, estrus': [11, 21, 14, 27, 37, 67, 23, 59, 21, 46, 28],
    'F-M, diestrus': [16, 24, 39, 15, 13, 24, 21, 15, 28, 27, 29, 23],
    'F-F, ECC, estrus': [28, 30, 38, 33, 19, 22, 14, 30],
    'F-F, ECC, diestrus': [10, 22, 13, 30, 21, 24, 9, 25, 44],
    'F-F, estrus': [25, 19, 26, 18, 28, 39, 21, 32, 19],
    'F-F, diestrus': [10, 22, 20, 26, 34, 28, 13, 24, 38,],
}

interaction_times_dictionary_NAc = {
    'control switch': [16, 10, 2, 8, 3, 5, 6, 12, 2,],
    'novel floor': [17, 25, 46, 39, 70, 16, 15, 11, 7,],
    'novel object': [3, 3, 6, 19, 16, 12, 4, 10,],
    'male meets male': [],
    'male meets female': [],
    'female meets male, estrus': [21, 9, 35, 13, 12, 16, 17, 21, 19, 7,],
    'female meets male, diestrus': [51, 14, 14, 21, 36, 9, 17, 25, 21, 22,],
    'female meets female, estrus': [22, 25, 12, 17, 8, 12, 35,],
    'female meets female, diestrus': [8, 29, 32, 13, 39, 36, 15,],
}

control_interaction_times_dictionary_NAc = {
    'control switch': [12, 7, 5, 3, 4, 8, 5, 3, 4,],
    'novel floor': [6, 4, 38, 9, 100, 5, 4, 9, 4,],
    'novel object': [3, 1, 3, 8, 3, 5, 7, 3,],
    'male meets male': [],
    'male meets female': [],
    'female meets male, estrus': [2, 3, 5, 8, 4, 11, 6, 3, 7, 2,],
    'female meets male, diestrus': [8, 10, 5, 9, 6, 5, 5, 8, 3, 7,],
    'female meets female, estrus': [4, 6, 3, 7, 5, 8, 4,],
    'female meets female, diestrus': [5, 6, 12, 4, 3, 5, 4,],
}

if remove_outliers:
    dicts = [interaction_times_dictionary_PFC, revisit_times_dictionary_PFC]
    dict_names = ['interaction times PFC', 'revisit times PFC']

    merged_datapoints = []
    merged_keys = []
    merged_dict_names = []

    for i, dict in enumerate(dicts):
        for key in dict.keys():
            for datapoint in dict[key]:
                merged_datapoints.append(datapoint)
                merged_keys.append(key)
                merged_dict_names.append(dict_names[i])

    from outliers import smirnov_grubbs as grubbs

    outlier_list = grubbs.two_sided_test_outliers(merged_datapoints, alpha=outlier_alpha)

    print('Outliers identified:', outlier_list)

    for i in range(len(outlier_list)):
        outlier_index = merged_datapoints.index(outlier_list[i])
        outlier_key = merged_keys[outlier_index]
        outlier_dict = dicts[dict_names.index(merged_dict_names[outlier_index])]

        outlier_dict[outlier_key].remove(outlier_list[i])


def average_over_repetitions(data1, data2, ID1, ID2):
    all_IDs = list(set(ID1) | set(ID2))
    averaged_data = []
    for ID in all_IDs:
        total = 0
        count = 0
        if ID in ID1:
            id_index = ID1.index(ID)
            total += data1[id_index]
            count += 1
        if ID in ID2:
            id_index = ID2.index(ID)
            total += data2[id_index]
            count += 1
        if count > 0:
            averaged_data.append(total/count)
    return averaged_data

cage_estrus = average_over_repetitions(
    interaction_times_dictionary_PFC['F-M, ECC, estrus'],
    interaction_times_dictionary_PFC['F-F, ECC, estrus'],
    animal_list_for_empty_cage_experiments['F-M, ECC, estrus'],
    animal_list_for_empty_cage_experiments['F-F, ECC, estrus'],
)

cage_diestrus = average_over_repetitions(
    interaction_times_dictionary_PFC['F-M, ECC, diestrus'],
    interaction_times_dictionary_PFC['F-F, ECC, diestrus'],
    animal_list_for_empty_cage_experiments['F-M, ECC, diestrus'],
    animal_list_for_empty_cage_experiments['F-F, ECC, diestrus'],
)

cage_estrus_revisit = average_over_repetitions(
    revisit_times_dictionary_PFC['F-M, ECC, estrus'],
    revisit_times_dictionary_PFC['F-F, ECC, estrus'],
    revisit_times_dictionary_PFC['F-M, ECC, estrus'],
    revisit_times_dictionary_PFC['F-F, ECC, estrus'],
)
cage_diestrus_revisit = average_over_repetitions(
    revisit_times_dictionary_PFC['F-M, ECC, diestrus'],
    revisit_times_dictionary_PFC['F-F, ECC, diestrus'],
    revisit_times_dictionary_PFC['F-M, ECC, diestrus'],
    revisit_times_dictionary_PFC['F-F, ECC, diestrus'],
)

cage_estrus.sort()
cage_diestrus.sort()


if combine_combine_female_cage_interactions:
    del interaction_times_dictionary_PFC['F-M, ECC, estrus']
    del interaction_times_dictionary_PFC['F-F, ECC, estrus']
    del interaction_times_dictionary_PFC['F-M, ECC, diestrus']
    del interaction_times_dictionary_PFC['F-F, ECC, diestrus']

    interaction_times_dictionary_PFC['F, ECC, estrus'] = cage_estrus
    interaction_times_dictionary_PFC['F, ECC, diestrus'] = cage_diestrus

    del revisit_times_dictionary_PFC['F-M, ECC, estrus']
    del revisit_times_dictionary_PFC['F-F, ECC, estrus']
    del revisit_times_dictionary_PFC['F-M, ECC, diestrus']
    del revisit_times_dictionary_PFC['F-F, ECC, diestrus']

    revisit_times_dictionary_PFC['F, ECC, estrus'] = cage_estrus_revisit
    revisit_times_dictionary_PFC['F, ECC, diestrus'] = cage_diestrus_revisit


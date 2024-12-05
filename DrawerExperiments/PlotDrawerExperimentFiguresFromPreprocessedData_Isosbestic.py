# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Produces the summary plots for the isosbestic signal as a function of the stimulus inserted in the drawer

import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import utils.statsfunctions as statfunctions
import scipy.stats as stats

with open('../PreprocessedData/CompleteDrawerExperimentDataset.obj', "rb") as input_file:
    experiments = pickle.load(input_file)

experiment_keys = [
                        'Drawer_Control_Swap',
                        'Drawer_Novel_Floor',
                        'Drawer_Novel_Object',
                        'Drawer_M_EC',
                        'Drawer_M_F',
                        'Drawer_M_M',

                        'Drawer_F_F_Diestrus', # Get the cage visits from the female interaction experiments
                        'Drawer_F_F_Estrus',
                        'Drawer_F_M_Diestrus',
                        'Drawer_F_M_Estrus',

                        'Drawer_F_F_Diestrus', # Get the mouse visits from the female interaction experiments
                        'Drawer_F_F_Estrus',
                        'Drawer_F_M_Diestrus',
                        'Drawer_F_M_Estrus',
                        ]

events = [
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],

    ['stimulus inserted (empty cage)', 'drawer entered after stimulus insertion (empty cage)', 'drawer revisited after stimulus removal (empty cage)'],
    ['stimulus inserted (empty cage)', 'drawer entered after stimulus insertion (empty cage)', 'drawer revisited after stimulus removal (empty cage)'],
    ['stimulus inserted (empty cage)', 'drawer entered after stimulus insertion (empty cage)', 'drawer revisited after stimulus removal (empty cage)'],
    ['stimulus inserted (empty cage)', 'drawer entered after stimulus insertion (empty cage)', 'drawer revisited after stimulus removal (empty cage)'],

    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
    ['stimulus inserted', 'drawer entered after stimulus insertion', 'drawer revisited after stimulus removal'],
]

dataset_names = [
    'cont.',
    'floor',
    'object',
    'M-Cage',
    'M-M',
    'M-F',

    'F-F d cage',
    'F-F e cage',
    'F-M d cage',
    'F-M e cage',

    'F-F d',
    'F-F e',
    'F-M d',
    'F-M e']

generic_event_names = ['enter', 'visit', 'empty']

def get_windows_from_experiment(experiment, events):




    dataset = experiment['data']
    IDs = []
    for data in dataset:
        IDs.append(data['Mouse ID'])

    number_of_animals = len(dataset)

    def frame_to_time(d, frame):
        return (frame - d['red light'][0])/30.0


    def get_windows(data, window_size_s, events):
        F = data['dF/F Isosbestic']
        ts = data['times']
        window_size_index_L = int(window_size_s[0] * data['fs'])
        window_size_index_R = int(window_size_s[1] * data['fs'])
        window_ts = None
        z = F

        windows = []
        for event in events:
            try:
                event_item = data[event]
            except:
                print('unknown event =', event, event_item)
                print(list(data.keys()))
                raise Exception('Unknown key')
            if type(event_item) != list:
                event_item = [event_item]
            if len(event_item) > 0:
                t_event = frame_to_time(data, event_item[0])
                event_index = np.argmin(np.square(ts - t_event))
                t_0 = ts[event_index]
                window = z[event_index + window_size_index_L: event_index + window_size_index_R + 1]
                if window_ts is None:
                    window_ts = ts[event_index + window_size_index_L: event_index + window_size_index_R + 1]
                    window_ts = window_ts - t_0
                n = event_index + window_size_index_R + 1 - (event_index + window_size_index_L)
                if window.shape[0] < n:
                    window = np.concatenate([window, np.NaN * np.zeros((n - window.shape[0]))], axis=0)

                if np.sum(np.isnan(window[np.logical_and(window_ts >0, window_ts < 5)])) > 0:
                    window *= np.nan

                windows.append(window)

            else:
                n = window_size_index_R + 1 - window_size_index_L
                window = np.zeros((n,)) * np.NaN
                windows.append(window)


        windows = np.array(windows)
        return window_ts, windows


    windows = []
    ts = None
    for animal_number in range(number_of_animals):
        ts, w = get_windows(dataset[animal_number], [-20, 20], events)
        windows.append(w)
    windows = np.array(windows)
    return ts, windows, IDs


axis_with_plots = []
female_meets_male = []
male_meets_X = []
perievent_data = []

for condition_number, experiment_key in enumerate(experiment_keys):

    ts, windows, IDs = get_windows_from_experiment(experiment=experiments[experiment_key], events=events[condition_number])
    perievent_fluorescence_baselined = windows * 0
    for i, event in enumerate(events[condition_number]):
        perievent_fluorescence_baselined[:, i, :] = 100*(windows[:, i, :] - np.mean(windows[:, i, ts < -8], axis=1, keepdims=True))

    perievent_data.append({
        'name': dataset_names[condition_number],
        'ts': ts,
        'F': perievent_fluorescence_baselined,
        'IDs': IDs,
    })


def average_recordings(data_list: list, name_list:list, name_1: str, name_2: str, new_name: str):

    index_1 = name_list.index(name_1)
    index_2 = name_list.index(name_2)

    x: dict = data_list[index_1]
    y: dict = data_list[index_2]

    all_IDs = list(set(x['IDs']) | set(y['IDs']))

    averaged_data = []
    for ID in all_IDs:
        F_sum= 0
        count = 0
        if ID in x['IDs']:
            id_index = x['IDs'].index(ID)
            F_sum += x['F'][id_index, :, :]
            count += 1
        if ID in y['IDs']:
            id_index = y['IDs'].index(ID)
            F_sum += y['F'][id_index, :, :]
            count += 1
        if count > 0:
            averaged_data.append(F_sum / count)

    averaged_data = np.array(averaged_data)

    averaged_data_dict = {
        'name': new_name,
        'ts': x['ts'],
        'F': averaged_data,
        'IDs': all_IDs,
    }

    if index_1 < index_2:

        del data_list[index_2]
        del data_list[index_1]

        del name_list[index_2]
        del name_list[index_1]

        data_list.insert(index_1, averaged_data_dict)
        name_list.insert(index_1, new_name)
    else:
        del data_list[index_1]
        del data_list[index_2]

        del name_list[index_1]
        del name_list[index_2]

        data_list.insert(index_2, averaged_data_dict)
        name_list.insert(index_2, new_name)


    # 'F-F d cage',
    # 'F-F e cage',
    # 'F-M d cage',
    # 'F-M e cage',

average_recordings(perievent_data, dataset_names, 'F-F d cage', 'F-M d cage', 'F-cage d')
average_recordings(perievent_data, dataset_names, 'F-F e cage', 'F-M e cage', 'F-cage e')


# some recordings of the revisits do not include the full 20 seconds. These can be plotted here
# Any dataset not including 0-5 seconds after the beginning of the revisit was dropped from the analysis

def combine_estrus_and_diestrus_dicts_for_plot(data_list, name_list, estrus_name, diestrus_name, new_name):
    index_1 = name_list.index(estrus_name)
    index_2 = name_list.index(diestrus_name)

    x: dict = data_list[index_1]
    y: dict = data_list[index_2]

    new_dict = {
        'name': new_name,
        'ts': x['ts'],
        'F': [x['F'], y['F']],
        'IDs': [x['IDs'], y['IDs']],
    }

    if index_1 < index_2:

        del data_list[index_2]
        del data_list[index_1]

        del name_list[index_2]
        del name_list[index_1]

        data_list.insert(index_1, new_dict)
        name_list.insert(index_1, new_name)
    else:
        del data_list[index_1]
        del data_list[index_2]

        del name_list[index_1]
        del name_list[index_2]

        data_list.insert(index_2, new_dict)
        name_list.insert(index_2, new_name)

combine_estrus_and_diestrus_dicts_for_plot(perievent_data, dataset_names, 'F-F e', 'F-F d', 'F-F')
combine_estrus_and_diestrus_dicts_for_plot(perievent_data, dataset_names, 'F-M e', 'F-M d', 'F-M')
combine_estrus_and_diestrus_dicts_for_plot(perievent_data, dataset_names, 'F-cage e', 'F-cage d', 'F-cage')



# fig, axes = plt.subplots(len(events[0]), len(perievent_data), figsize=(11, 5))
# plt.subplots_adjust(bottom=0.1, left=0.25, wspace=0.3, hspace=1)
#
# for condition_number, perievent_data_dict in enumerate(perievent_data):
#     if type(perievent_data_dict['F']) == list:
#         number_of_events = perievent_data_dict['F'][0].shape[1]
#     else:
#         number_of_events = perievent_data_dict['F'].shape[1]
#     for event_index in range(number_of_events):
#         axis = axes[event_index][condition_number]
#         if event_index == 0:
#             axis.set_title(perievent_data_dict['name'])
#         ts = perievent_data_dict['ts']
#
#         if type(perievent_data_dict['F']) == list:
#             color_list = [prettyplot.colors['red'], prettyplot.colors['blue']]
#             faint_color_list = [[1, 0.8, 0.8], [0.8, 0.8, 1]]
#             for estrus_index, F in enumerate(perievent_data_dict['F']):
#                 w_baselined = F[:, event_index, :]
#
#                 m = np.nanmean(w_baselined, axis=0)
#                 sem = np.nanstd(w_baselined, axis=0) / np.sqrt(w_baselined.shape[0])
#
#                 axis.fill_between(ts, m - sem, m + sem, color=faint_color_list[estrus_index])
#                 axis.plot(ts, m, color=color_list[estrus_index])
#         else:
#             w_baselined = perievent_data_dict['F'][:, event_index, :]
#
#             m = np.nanmean(w_baselined, axis=0)
#             sem = np.nanstd(w_baselined, axis=0)/np.sqrt(w_baselined.shape[0])
#             if perievent_data_dict['name'][0] == 'M':
#                 color = prettyplot.colors['green']
#                 faint_color = [0.7, 1, 0.7]
#             else:
#                 color = prettyplot.colors['black']
#                 faint_color = [0.8, 0.8, 0.8]
#             axis.fill_between(ts, m - sem, m + sem, color=faint_color)
#             axis.plot(ts, m, color=color)
#
#         axis.set_ylim([-3, 15])
#         prettyplot.xlabel('t s', axis)
#         if condition_number == 0:
#             prettyplot.no_box(axis)
#             prettyplot.ylabel('dF/F', axis)
#         else:
#             prettyplot.x_axis_only(axis)
#
#         axis_with_plots.append((condition_number, i))
#
#
# plt.savefig('FigurePdfs/IsosbesticPlots/PerieventAveragesForAllExperiments.pdf', transparent=True)
# plt.show()


def get_groups(data_list, name_list, condition_names, which_event=1, remove_nans=False):

    Fs = []

    for name in condition_names:
        if ' estrus' in name and not ('diestrus' in name):
            name_shortened = name[:-len(' estrus')]
            index = name_list.index(name_shortened)
            f = data_list[index]['F'][0][:, which_event, :]
        elif ' diestrus' in name:
            name_shortened = name[:-len(' diestrus')]
            index = name_list.index(name_shortened)
            f = data_list[index]['F'][1][:, which_event, :]
        else:
            index = name_list.index(name)
            f = data_list[index]['F'][:, which_event, :]

        f = np.nanmean(f[:, np.logical_and(data_list[index]['ts'] >= 0, data_list[index]['ts'] <= 5)], axis=1)

        if remove_nans:
            I = np.isnan(f)
            f = np.delete(f, I)

        Fs.append(f)
    return Fs


group_names = [
    'cont.',
    'floor',
    'object',
    'M-Cage',
    'M-M',
    'M-F',
    'F-cage diestrus',
    'F-F diestrus',
    'F-M diestrus',
    'F-cage estrus',
    'F-F estrus',
    'F-M estrus'
]

colors = []
faint_colors = []
for name in group_names:
    if 'diestrus' in name:
        colors.append(prettyplot.colors['blue'])
        faint_colors.append([0.8, 0.8, 1])
    elif 'estrus' in name:
        colors.append(prettyplot.colors['red'])
        faint_colors.append([1, 0.8, 0.8])
    elif name[0] == 'M':
        colors.append(prettyplot.colors['green'])
        faint_colors.append([0.8, 1, 0.8])
    else:
        colors.append(prettyplot.colors['black'])
        faint_colors.append([0.8, 0.8, 0.8])

groups = get_groups(perievent_data, dataset_names, group_names)
means = np.array([np.nanmean(g) for g in groups])
sems = np.array([np.nanstd(g)/np.sqrt(g.shape[0]) for g in groups])


fig = plt.figure(figsize=(4, 2))
plt.subplots_adjust(bottom=0.25)
current_position = 0
large_width = 1
small_width = 0.33
positions = []
for i, g in enumerate(groups):
    positions.append(current_position)
    plt.scatter(current_position * np.ones(shape=(len(g),)), g, color=faint_colors[i])
    plt.errorbar(current_position, means[i], yerr=sems[i], capsize=5, color=colors[i], linestyle='', marker='o')

    if 'diestrus' in group_names[i]:
        current_position += large_width
    elif 'estrus' in group_names[i]:
        current_position += large_width
    else:
        current_position += large_width

plt.xticks(positions, group_names, rotation=45)
for tick in plt.gca().xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")
prettyplot.no_box()

control_names = [
    None,
    'cont.',
    'cont.',
    None,
    'M-Cage',
    'M-Cage',
    None,
    'F-cage diestrus',
    'F-cage diestrus',
    None,
    'F-cage estrus',
    'F-cage estrus'
]

def get_t_tests_from_respective_controls(groups, names, control_names):
    pvals = [None]*len(names)
    unique_control_names = set(control_names)
    unique_control_names.remove(None)

    for control_name in unique_control_names:
        control_index = names.index(control_name)
        indices = [i for i, n in enumerate(names) if control_names[i] == control_name]
        group_pvals = []

        y = [groups[indices[j]] for j in range(len(indices))]
        y = y + [groups[control_index]]
        F_statistic, pvalue = statfunctions.ANOVA_one_way(y, equal_variances=False)
        pairwise_pvals, pairs = statfunctions.multi_t_test_bonferroni_corrected(y)

        print('')
        print('group:', [control_name, names[indices[0]], names[indices[1]] ])
        print('one way ANOVA:', 'F =', F_statistic, 'pvalue =', pvalue, 'n =', [x.shape[0] for x in y])
        if pvalue <= 0.05:
            print('AllPairwise: ' + control_name, pairwise_pvals)

        for j in range(len(indices)):
            group_pvals.append(stats.ttest_ind(groups[indices[j]], groups[control_index], equal_var=False).pvalue)

        group_pvals = statfunctions.apply_bonferroni_correction(group_pvals)

        for j in range(len(indices)):
            pvals[indices[j]] = group_pvals[j]

    return pvals

pvals = get_t_tests_from_respective_controls(groups, group_names, control_names)
print('')


star_height = 25
for i, p in enumerate(pvals):
    if pvals[i] is not None:
        if pvals[i] < 0.01:
            plt.text(positions[i], star_height, '**', fontsize=18, horizontalalignment='center',
                     verticalalignment='center')
        elif pvals[i] < 0.05:
            plt.text(positions[i], star_height, '*', fontsize=18, horizontalalignment='center',
                     verticalalignment='center')
        else:
            plt.text(positions[i], star_height, 'N.S.', fontsize=12, horizontalalignment='center',
                     verticalalignment='center')

ylim = [-5, 22]
plt.ylim(ylim)

plt.savefig('FigurePdfs/IsosbesticPlots/AverageResponseToDrawerStimulus.pdf', transparent=True)

plt.show()

#######################################################################
##### Same plot for the revisit
#######################################################################


group_names = [
    'cont.',
    'floor',
    'object',
    'M-Cage',
    'M-M',
    'M-F',
    'F-cage diestrus',
    'F-F diestrus',
    'F-M diestrus',
    'F-cage estrus',
    'F-F estrus',
    'F-M estrus'
]

colors = []
faint_colors = []
for name in group_names:
    if 'diestrus' in name:
        colors.append(prettyplot.colors['blue'])
        faint_colors.append([0.8, 0.8, 1])
    elif 'estrus' in name:
        colors.append(prettyplot.colors['red'])
        faint_colors.append([1, 0.8, 0.8])
    elif name[0] == 'M':
        colors.append(prettyplot.colors['green'])
        faint_colors.append([0.8, 1, 0.8])
    else:
        colors.append(prettyplot.colors['black'])
        faint_colors.append([0.8, 0.8, 0.8])

# In some F-X experiments the recording was stopped early so there is not a complete window. These
# datapoints have been removed.
groups = get_groups(perievent_data, dataset_names, group_names, which_event=2, remove_nans=True)
means = np.array([np.nanmean(g) for g in groups])
sems = np.array([np.nanstd(g)/np.sqrt(g.shape[0]) for g in groups])

fig = plt.figure(figsize=(4, 2))
plt.subplots_adjust(bottom=0.25)
current_position = 0
large_width = 1
small_width = 0.33
positions = []
for i, g in enumerate(groups):
    positions.append(current_position)
    plt.scatter(current_position * np.ones(shape=(len(g),)), g, color=faint_colors[i])
    plt.errorbar(current_position, means[i], yerr=sems[i], capsize=5, color=colors[i], linestyle='', marker='o')

    if 'diestrus' in group_names[i]:
        current_position += large_width
    elif 'estrus' in group_names[i]:
        current_position += large_width
    else:
        current_position += large_width

plt.xticks(positions, group_names, rotation=45)
for tick in plt.gca().xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")
prettyplot.no_box()

control_names = [
    None,
    'cont.',
    'cont.',
    None,
    'M-Cage',
    'M-Cage',
    None,
    'F-cage diestrus',
    'F-cage diestrus',
    None,
    'F-cage estrus',
    'F-cage estrus'
]

def get_t_tests_from_respective_controls(groups, names, control_names):
    pvals = [None]*len(names)
    unique_control_names = set(control_names)
    unique_control_names.remove(None)

    for control_name in unique_control_names:
        control_index = names.index(control_name)
        indices = [i for i, n in enumerate(names) if control_names[i] == control_name]
        group_pvals = []

        y = [groups[indices[j]] for j in range(len(indices))]
        y = y + [groups[control_index]]
        F_statistic, pvalue = statfunctions.ANOVA_one_way(y, equal_variances=False)
        pairwise_pvals, pairs = statfunctions.multi_t_test_bonferroni_corrected(y)

        print('')
        print('group (REVISIT):', [control_name, names[indices[0]], names[indices[1]] ])
        print('one way ANOVA:', 'F =', F_statistic, 'pvalue =', pvalue, 'n =', [x.shape[0] for x in y])
        if pvalue <= 0.05:
            print('AllPairwise: ' + control_name, pairwise_pvals)

        for j in range(len(indices)):
            group_pvals.append(stats.ttest_ind(groups[indices[j]], groups[control_index], equal_var=False).pvalue)

        group_pvals = statfunctions.apply_bonferroni_correction(group_pvals)

        for j in range(len(indices)):
            pvals[indices[j]] = group_pvals[j]

    return pvals

pvals = get_t_tests_from_respective_controls(groups, group_names, control_names)



star_height = 25
for i, p in enumerate(pvals):
    if pvals[i] is not None:
        if pvals[i] < 0.01:
            plt.text(positions[i], star_height, '**', fontsize=18, horizontalalignment='center',
                     verticalalignment='center')
        elif pvals[i] < 0.05:
            plt.text(positions[i], star_height, '*', fontsize=18, horizontalalignment='center',
                     verticalalignment='center')
        else:
            plt.text(positions[i], star_height, 'N.S.', fontsize=12, horizontalalignment='center',
                     verticalalignment='center')
plt.ylim(ylim)
plt.savefig('FigurePdfs/IsosbesticPlots/AverageResponseToDrawerReentry.pdf', transparent=True)
plt.show()




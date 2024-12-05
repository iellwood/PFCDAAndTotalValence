import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import scipy.stats as stats
import utils.statsfunctions as statsfunctions

with open('CompleteWaterExperimentData.obj', "rb") as input_file:
    experiment = pickle.load(input_file)

left_pad = 0.4

datasets = experiment['data']

def get_peri_event(ts, x, event_times, window_size, include_partial_windows=False):
    '''
    Returns data around event_times in a window of size 2*window_size + 1.
    Note that this algorithm assumes that ts is a uniform step size list of times.

    :param ts: 1d list of times. Must be same length as x
    :param x: 1d data. Must be same length as ts
    :param event_times: 1d list of events
    :param window_size:
    :return:
    '''


    # step_size
    dt = ts[1] - ts[0]

    if type(window_size) == tuple or type(window_size) == list:
        n_left = int(np.floor(window_size[0] / dt))
        n_right = int(np.floor(window_size[1] / dt))
    else:
        n_left = int(np.floor(window_size / dt))
        n_right = int(np.floor(window_size / dt))


    m = len(x)

    peri_event_list = []

    for i in range(event_times.shape[0]):
        event_t = event_times[i]
        event_index = np.argmin(np.abs(ts - event_t))  # time index closest to event

        if 0 <= event_index < m:  # make sure event is inside data

            left_index = np.maximum(event_index - n_left, 0)
            left_pad = left_index - (event_index - n_left)

            right_index = np.minimum(event_index + n_right, m - 1)
            right_pad = (event_index + n_right) - right_index

            y = x[left_index:(right_index + 1)]
            y = np.pad(y, (left_pad, right_pad), 'constant', constant_values=(np.nan, np.nan))

            if include_partial_windows or (right_pad == 0 and left_pad == 0):
                peri_event_list.append(np.expand_dims(y, 1))

    ts_window = ts[:(n_left + n_right + 1)]
    ts_window = ts_window - ts_window[n_left]

    if len(peri_event_list) > 0:
        return ts_window, np.concatenate(peri_event_list, axis=1)
    else:
        return ts_window, None


peri_event_water_data = []
peri_event_no_water_data = []
peri_event_beam_breaks_data = []



for d in datasets:
    F = d['dF/F Excitation after isosbestic subtraction']  # excitation

    ts = d['times']
    water_events = d['Event times: water delivered']
    no_water_events = d['Event times: water omitted']
    beam_breaks = d['Event times: beam break']

    ts_window, peri_data = get_peri_event(ts, F, water_events, (10, 20))
    ts_window, peri_no_water_data = get_peri_event(ts, F, no_water_events, (10, 20))

    ts_window, peri_data_beam_break = get_peri_event(ts, F, beam_breaks, (10, 20))


    peri_event_water_data.append(np.mean(peri_data, 1))
    peri_event_no_water_data.append(np.mean(peri_no_water_data, 1))
    peri_event_beam_breaks_data.append(np.mean(peri_data_beam_break, 1))

peri_event_water_data = np.array(peri_event_water_data)
peri_event_no_water_data = np.array(peri_event_no_water_data)
peri_event_beam_breaks_data = np.array(peri_event_beam_breaks_data)


# baseline the data over the window [-5, 0]
peri_event_water_data = peri_event_water_data - np.mean(peri_event_water_data[:, ts_window < -5], 1, keepdims=True)
peri_event_no_water_data = peri_event_no_water_data - np.mean(peri_event_no_water_data[:, ts_window < -5], 1, keepdims=True)
peri_event_beam_breaks_data = peri_event_beam_breaks_data - np.mean(peri_event_beam_breaks_data[:, ts_window < -5], 1, keepdims=True)



peri_event_water_data *= 100
peri_event_no_water_data *= 100
peri_event_beam_breaks_data *= 100

FiveSecondWindowMask = np.logical_and(ts_window >= 0, ts_window <= 5)

water_averages = np.mean(peri_event_water_data[:, FiveSecondWindowMask], 1)
no_water_averages = np.mean(peri_event_no_water_data[:, FiveSecondWindowMask], 1)
beam_break_averages = np.mean(peri_event_beam_breaks_data[:, FiveSecondWindowMask], 1)



mean_water = np.mean(peri_event_water_data, 0)
sem_water = np.std(peri_event_water_data, 0)/np.sqrt(peri_event_water_data.shape[0])

mean_no_water = np.mean(peri_event_no_water_data, 0)
sem_no_water = np.std(peri_event_no_water_data, 0)/np.sqrt(peri_event_no_water_data.shape[0])

mean_beam_break = np.mean(peri_event_beam_breaks_data, 0)
sem_beam_break = np.std(peri_event_beam_breaks_data, 0)/np.sqrt(peri_event_beam_breaks_data.shape[0])


I = ts_window >= -5

# Plot the perievent averages around the first lick
fig = plt.figure(figsize=(2, 2))
plt.subplots_adjust(bottom=0.2, left=left_pad)

plt.fill_between(ts_window[I], mean_water[I] - sem_water[I], mean_water[I] + sem_water[I], color=[0.8, 0.8, 1])
plt.fill_between(ts_window[I], mean_no_water[I] - sem_no_water[I], mean_no_water[I] + sem_no_water[I], color=[0.8, 0.8, 0.8])

plt.plot(ts_window[I], mean_water[I], color=prettyplot.colors['blue'])
plt.plot(ts_window[I], mean_no_water[I], color=prettyplot.colors['red'])
prettyplot.no_box()
plt.ylim([-0.5, 1.5])
plt.savefig('FigurePdfs/WaterVsNoWater_PerieventAverage.pdf', transparent=True)
plt.show()

# Plot the perievent averages around the beam break
fig = plt.figure(figsize=(2, 2))
plt.subplots_adjust(bottom=0.2, left=left_pad)
plt.fill_between(ts_window[I], mean_beam_break[I] - sem_beam_break[I], mean_beam_break[I] + sem_beam_break[I], color=[0.8, 0.8, 0.8])
plt.plot(ts_window[I], mean_beam_break[I], color='k')
prettyplot.no_box()
plt.ylim([-0.5, 1.5])

plt.savefig('FigurePdfs/BeamBreak_PerieventAverage.pdf', transparent=True)
plt.show()

plt.figure(figsize=(1, 2))
plt.subplots_adjust(left=0.3, bottom=0.2)
plt.scatter(np.ones(shape=beam_break_averages.shape) * 0, beam_break_averages, color=[0.8, 0.8, 0.8], clip_on=False)
plt.scatter(np.ones(shape=no_water_averages.shape) * 1, no_water_averages, color=[0.8, 0.8, 0.8], clip_on=False)
plt.scatter(np.ones(shape=water_averages.shape) * 2, water_averages, color=[0.8, 0.8, 0.8], clip_on=False)
plt.errorbar(
    [0, 1, 2],
    [np.mean(beam_break_averages), np.mean(no_water_averages), np.mean(water_averages)],
    yerr = [np.std(beam_break_averages)/np.sqrt(beam_break_averages.shape[0]),
     np.std(no_water_averages)/np.sqrt(no_water_averages.shape[0]),
     np.std(water_averages)/np.sqrt(water_averages.shape[0]),],
    color='k',
    marker='o',
    capsize=5,
    linestyle='',
    clip_on=False
)
plt.xticks([0, 1, 2], ['BB', 'O', 'W'])
prettyplot.no_box()
plt.xlim([-0.5, 2.25])
prettyplot.ylabel('%dF/F')

joint_data = np.concatenate([beam_break_averages[None, :], no_water_averages[None, :], water_averages[None, :]], 0)

p, F = statsfunctions.repeated_measures_ANOVA(np.array([0, 1, 2]), joint_data)

print('Repeated Measures ANOVA:', 'p =', p, 'F =', F)

print('FearConditioningData shape =', joint_data.shape)


data_averages = [beam_break_averages, no_water_averages, water_averages]
pvals = []

for d in data_averages:

    s = stats.ttest_rel(d, d * 0)
    print(s)
    pvals.append(s.pvalue)

pvals = np.array(pvals)
print(pvals)

pvals = statsfunctions.apply_bonferroni_correction(pvals)
print('[beam break, no_water, water].pvalue =', pvals)

for i in range(3):
    if pvals[i] < 0.05:
        plt.text(i, 2, '*')
    else:
        plt.text(i, 2, 'N.S.')
plt.savefig('FigurePdfs/SummaryPlot.pdf', transparent=True)
plt.show()



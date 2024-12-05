import numpy as np
import matplotlib.pyplot as plt
import OriginalData.Data.FreezingData as freezing_data
import utils.prettyplot as prettyplot
def plot_freezing_data(day_1, day_2, figure_pdf_file_name):

    conditioning = day_1[:, :8]
    extinction = day_1[:, 8:]
    extinction_memory = day_2


    def average_over_groups_of_trials(data, width):
        s = data.shape
        data = np.reshape(data, [s[0], s[1]//width, width])
        data = np.mean(data, 2)
        return data


    conditioning = average_over_groups_of_trials(conditioning, 2)
    extinction= average_over_groups_of_trials(extinction, 2)
    extinction_memory = average_over_groups_of_trials(extinction_memory, 2)

    x_cond = np.arange(conditioning.shape[1])
    x_cond_tick_labels = np.arange(conditioning.shape[1], dtype=int) * 2 + 2


    x_ext = np.arange(extinction.shape[1]) + x_cond[-1] + 4
    x_ext_tick_labels = np.arange(extinction.shape[1], dtype=int) * 2 + 2

    x_em = np.arange(extinction_memory.shape[1]) + x_ext[-1] + 4
    x_em_tick_labels = np.arange(extinction_memory.shape[1], dtype=int) * 2 + 2

    mean_cond = np.mean(conditioning, 0)/5 * 100
    sem_cond = np.std(conditioning, 0)/np.sqrt(conditioning.shape[0])/5 * 100

    mean_ext = np.mean(extinction, 0)/5 * 100
    sem_ext = np.std(extinction, 0)/np.sqrt(extinction.shape[0])/5 * 100

    mean_em = np.mean(extinction_memory, 0)/5 * 100
    sem_em = np.std(extinction_memory, 0)/np.sqrt(extinction_memory.shape[0])/5 * 100

    fig = plt.figure(figsize=(5, 2))

    plt.errorbar(x_cond, mean_cond, yerr=sem_cond, capsize=5, marker='o', color='k', linestyle='')
    plt.errorbar(x_ext, mean_ext, yerr=sem_ext, capsize=5, marker='o', color='k', linestyle='')
    plt.errorbar(x_em, mean_em, yerr=sem_em, capsize=5, marker='o', color='k', linestyle='')
    prettyplot.no_box()

    plt.xticks(
        np.concatenate([x_cond, x_ext, x_em], 0),
        np.concatenate([x_cond_tick_labels, x_ext_tick_labels, x_em_tick_labels], 0)
    )
    plt.ylim([0, 100])

    plt.savefig('FigurePdfs/' + figure_pdf_file_name + '.pdf', transparent=True)

    plt.show()

plot_freezing_data(freezing_data.mPFC_Day_1, freezing_data.mPFC_Day_2, 'FreezingPlot_mPFC')
plot_freezing_data(freezing_data.NAcCore_Day_1, freezing_data.NAcCore_Day_2, 'FreezingPlot_NAcCore')


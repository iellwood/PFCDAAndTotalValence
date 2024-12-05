# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Plots the heatmaps using the saved heatmap data in HeatmapData/


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils.prettyplot as prettyplot
import auxiliaryfunctions.transformtracking
from utils.heatmap import *
import matplotlib.collections as collections
import matplotlib.patches as patches


def plot_saved_heatmap(heatmap_data_file, condition):

    #np.savez('HeatmapData/' + condition +'_heatmapdata', images=images, masks=masks)
    heatmap_data = np.load(heatmap_data_file)

    images = heatmap_data['images']
    masks = heatmap_data['masks']
    average_width = heatmap_data['average_width']
    center = heatmap_data['center']
    arm_length = heatmap_data['arm_length']
    center_radius = heatmap_data['center_radius']


    image, count = make_average_from_images_and_masks(images, masks)
    fig = plt.figure(figsize=(4, 3))
    if np.max(count) > 3:
        im = np.transpose(np.where(count > 2, image, np.NAN), (1, 0))
    else:
        im = np.transpose(np.where(count > 0, image, np.NAN), (1, 0))

    print(condition + ': max %dF/F =', np.nanmax(100*im))
    print(condition + ': min %dF/F =', np.nanmin(100*im))
    plt.imshow(100*im, cmap=matplotlib.colormaps['plasma'], extent=[-1.1, 1.1, -1.1, 1.1])
    rect1 = patches.Rectangle((center[0] - arm_length, center[1] - center_radius), arm_length * 2, center_radius * 2,
                              edgecolor='k', fill=False)
    rect2 = patches.Rectangle((center[0] - center_radius, center[1] - arm_length), center_radius * 2, arm_length * 2,
                              edgecolor='k', fill=False)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    prettyplot.no_axes()
    plt.colorbar()
    plt.savefig('FigurePdfs/' + condition + '_EPMHeatMapAverageAcrossAllMice.pdf', transparent=True)
    plt.show()

plot_saved_heatmap('HeatmapData/EPM_mPFC_heatmapdata.npz', 'mPFC')
plot_saved_heatmap('HeatmapData/EPM_NAcCore_heatmapdata.npz', 'NAcCore')


# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# Plots the isosbestic heatmaps using the saved heatmap data in HeatmapData/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils.prettyplot as prettyplot
import auxiliaryfunctions.transformtracking
from utils.heatmap import *
import matplotlib.collections as collections
import matplotlib.patches as patches


def plot_saved_heatmap(heatmap_data_file, condition, f_range=None):

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

    if f_range is not None: # This is a cheesy way to get the right range, but it works
        im[0, 0] = f_range[0]/100.0
        im[0, 1] = f_range[1]/100.0
    plt.imshow(100*im, cmap=matplotlib.colormaps['plasma'], extent=[-1.1, 1.1, -1.1, 1.1])
    rect1 = patches.Rectangle((center[0] - arm_length, center[1] - center_radius), arm_length * 2, center_radius * 2,
                              edgecolor='k', fill=False)
    rect2 = patches.Rectangle((center[0] - center_radius, center[1] - arm_length), center_radius * 2, arm_length * 2,
                              edgecolor='k', fill=False)

    rect3 = patches.Rectangle((-arm_length-.2, arm_length-0.1-.005), .2, .3, fill=True, color='w')

    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.gca().add_patch(rect3)
    prettyplot.no_axes()
    plt.colorbar()
    plt.savefig('FigurePdfs/Isosbestic/' + condition + '_EPMHeatMapAverageAcrossAllMice_Isosbestic.pdf', transparent=True)
    plt.show()

plot_saved_heatmap('HeatmapData/Isosbestic/EPM_mPFC_isosbestic_heatmapdata.npz', 'mPFC', f_range=[-1.1457368142753785, 4.494689232980629])
plot_saved_heatmap('HeatmapData/Isosbestic/EPM_NAcCore_isosbestic_heatmapdata.npz', 'NAcCore', f_range=[-4.84383686963774, 9.44704893771247])


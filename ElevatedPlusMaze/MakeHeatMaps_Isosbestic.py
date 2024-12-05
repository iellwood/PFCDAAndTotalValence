import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils.prettyplot as prettyplot
from utils.heatmap import *
import matplotlib.patches as patches
import pickle

def make_heatmap_for_condition(experiment, image_half_width):

    dataset = experiment['data']


    condition = experiment['Experiment Name']

    images = []
    masks = []
    for i in range(len(dataset)):
        arm_width = dataset[i]['EPM Arm width']
        center = dataset[i]['EPM center']
        arm_length = dataset[i]['EPM Arm length']
        center_radius = dataset[i]['EPM Center radius']
        image, mask = make_heatmap(dataset[i], center, arm_length*1.1, image_half_width, arm_width/10, arm_width/5, isosbestic=True)
        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    print('Saving heat map:', condition)
    np.savez('HeatmapData/Isosbestic/' + condition +'_isosbestic_heatmapdata', images=images, masks=masks, average_width=arm_width, center=center, arm_length=arm_length, center_radius=center_radius)

    # image, count = make_average_from_images_and_masks(images, masks)
    # fig = plt.figure(figsize=(4, 3))
    # if np.max(count) > 3:
    #     im = np.transpose(np.where(count > 2, image, np.NAN), (1, 0))
    # else:
    #     im = np.transpose(np.where(count > 0, image, np.NAN), (1, 0))
    # plt.imshow(100*im, cmap=matplotlib.colormaps['plasma'], extent=[-1.1, 1.1, -1.1, 1.1])
    # rect1 = patches.Rectangle((center[0] - arm_length, center[1] - center_radius), arm_length * 2, center_radius * 2, edgecolor='k', fill=False)
    # rect2 = patches.Rectangle((center[0] - center_radius, center[1] - arm_length), center_radius * 2, arm_length * 2, edgecolor='k', fill=False)
    # plt.gca().add_patch(rect1)
    # plt.gca().add_patch(rect2)
    # prettyplot.no_axes()
    # plt.colorbar()
    # plt.savefig('FigurePdfs/Isosbestic/' + condition + '_EPMHeatMapAverageAcrossAllMice.pdf', transparent=True)
    # plt.show()

with open('../PreprocessedData/CompleteEPMDataset.obj', "rb") as input_file:
    EPM_Complete_Dataset = pickle.load(input_file)

make_heatmap_for_condition(EPM_Complete_Dataset['EPM_mPFC'], 100)
make_heatmap_for_condition(EPM_Complete_Dataset['EPM_NAcCore'], 100)
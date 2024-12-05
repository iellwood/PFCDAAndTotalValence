# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

import numpy as np
from alive_progress import alive_bar

def make_heatmap(data, center, width, image_width, sigma, distance_threshold, isosbestic=False):
    c = 1/(2*sigma*sigma)
    d2 = distance_threshold * distance_threshold
    image = np.zeros(shape=(image_width*2 + 1, image_width*2 + 1))
    mask = np.zeros(shape=(image_width*2 + 1, image_width*2 + 1))

    xs = np.linspace(center[0] - width, center[0] + width, image.shape[0])[:, None] * np.ones(shape=image.shape)
    ys = np.linspace(center[1] - width, center[1] + width, image.shape[1])[None, :] * np.ones(shape=image.shape)

    path_x = data['x_coordinate'][::10][:, None]
    path_y = data['y_coordinate'][::10][:, None]

    path = np.concatenate([path_x, path_y], 1)
    with alive_bar(image.shape[0], force_tty=True) as bar:
        for i in range(image.shape[0]):
            for j in range(image.shape[0]):

                X = np.array([xs[i, j], ys[i, j]])

                squared_distances = np.sum(np.square(path - X[None, :]), 1)
                min_squared_distance = np.min(squared_distances)
                if min_squared_distance >= d2:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1
                    squared_distances = squared_distances - np.min(squared_distances)

                    w = np.exp(-c*squared_distances)
                    w = w/np.sum(w)
                    if not isosbestic:
                        image[i, j] = np.sum(w * data['dF/F Excitation after isosbestic subtraction'][::10])
                    else:
                        image[i, j] = np.sum(w * data['dF/F Isosbestic'][::10])

            bar()

    return image, mask

def make_average_from_images_and_masks(images, masks):

    image = images[0, :, :] * 0
    count = images[0, :, :] * 0

    for i in range(images.shape[0]):
        image += images[i, :, :] * masks[i, :, :]
        count += masks[i, :, :]

    images_normalized = np.where(count > 0, image/count, 0)

    return images_normalized, count
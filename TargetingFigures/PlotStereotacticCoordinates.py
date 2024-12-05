# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# This script plots the coordinates for the implant locations using Paxinos and Franklin's 2001 atlas.
#
# Paxinos, George, and Keith B.J. Franklin. The mouse brain in stereotaxic coordinates: hard cover edition. Access Online via Elsevier, 2001
#
# The online tool by Matt Gaidica was used for comparing locations in histological slices with the atlas.
#
# Permission to use images from Paxinos & Franklin is given in the Preface to the atlas: "As authors, we give permission
# for the reproduction of any figure from the atlas in other publications, provided that the atlas is cited."
#
# The figures from the atlas have been modified to remove the grid lines and other irrelevant aspects of the images.

import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
from ListsOfStereotacticCoordinates import mPFC_coordinates, NacCore_coordinates
import os

images = []
APs = []

file_names = os.listdir('PaxinosImages/')
for file_name in file_names:
    images.append(plt.imread('PaxinosImages/' + file_name))
    APs.append(float(file_name[8:-4]))

APs = np.array(APs)


all_coordinates = np.concatenate([mPFC_coordinates, NacCore_coordinates], 0)

ap_coordinates = all_coordinates[:, 1]

closest_images = []
for i in range(ap_coordinates.shape[0]):
    closest_images.append(np.argmin(np.square(ap_coordinates[i] - APs)))
closest_images = np.array(closest_images)


# Coordinate system for the Paxinos image
x_0 = np.array([468.5, 84])
dx = -(559.5 - x_0[0])
dy = 175 - x_0[1]
d = np.array([dx, dy])

def plot_stereotactic_coordinates(xy, x_0, d, axis=None):
    z = xy * d + x_0
    if axis is None:
        axis = plt.gca()
    axis.scatter(z[:, 0], z[:, 1], color=prettyplot.colors['blue'], s=10)

fig, axes = plt.subplots(2, 4, figsize=(10, 7))
axis_list = []
for i in range(2):
    for j in range(4):
        axis_list.append(axes[i][j])
        prettyplot.no_axes(axes[i][j])

for i in range(len(images)):
    axis = axis_list[i]
    axis.imshow(images[i])
    I = closest_images == i
    coordinates_on_this_image = all_coordinates[I, :]
    plot_stereotactic_coordinates(coordinates_on_this_image[:, [0, 2]], x_0, d, axis)
    prettyplot.no_axes(axis)
    axis.set_xlim(115, x_0[0] + 1)
    axis.set_ylim(644, 122)
    axis.text(x_0[0] - 100, x_0[1], str(APs[i]))

plt.savefig('FigurePdfs/AllCoordinates.pdf', transparent=True, dpi=300)
plt.show()


print('Average PFC coordinate =', np.mean(mPFC_coordinates, 0))
print('Average NAc core coordinate =', np.mean(NacCore_coordinates, 0))

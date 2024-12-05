# Code from "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli"
# Authors: Y. Yang, W. Parent, H. Rice, R. Mark, M. Desimone, M. Rahman and I. T. Ellwood
# First uploaded version 12/5/2024
# Questions about the code should be directed to the corresponding author I.T. Ellwood

# List of all targeting coordinates estimated by comparing fixed slices with Paxinos & Franklin 2001
# Paxinos, George, and Keith B.J. Franklin. The mouse brain in stereotaxic coordinates: hard cover edition. Access Online via Elsevier, 2001
#
# The online tool by Matt Gaidica was used for comparing locations in histological slices with the atlas.
#
# Permission to use images from Paxinos & Franklin is given in the Preface to the atlas: "As authors, we give permission
# for the reproduction of any figure from the atlas in other publications, provided that the atlas is cited."
#
# The figures from the atlas have been modified to remove the grid lines and other irrelevant aspects of the images.


import numpy as np

mPFC_coordinates = [
    [0.6, 1.78, 2.55],
    [0.5, 1.7, 2.6],
    [0.4, 1.78, 2.4],
    [0.4, 1.78, 2.5],
    [0.3, 1.54, 2.7],
    [0.5, 1.65, 2.7],
    [0.45, 1.7, 2.6],
    [0.4, 1.7, 2.55],
    [0.35, 1.7, 2.45],
    [0.55, 1.54, 2.5],
    [0.5, 1.54, 2.65],
    [0.4, 1.54, 2.55],
    [0.5, 1.54, 2.7],
    [0.45, 1.7, 2.7],
    [0.6, 1.7, 2.5],
    [0.5, 1.7, 2.6],
    [0.55, 1.7, 2.45],
    [0.4, 1.78, 2.5],
    [0.5, 1.7, 2.5],
    [0.4, 1.54, 2.4],
    [0.5, 1.54, 2.6],
    [0.45, 1.54, 2.5],
    [0.5, 1.7, 2.55],
    [0.45, 1.7, 2.45],
    [0.5, 1.7, 2.5],
    [0.45, 1.78, 2.5],
    [0.5, 1.7, 2.4],
    [0.45, 1.7, 2.5],
    [0.55, 1.7, 2.6],
    [0.5, 1.7, 2.65],
    # [0, 0.98, 0],  # Uncomment these lines to place markers on the grid lines.
    # [0, 1.7, 0],
    # [0, 1.18, 0],
    # [0, 1.34, 0],
    # [0, 1.42, 0],
    # [0, 1.54, 0],
    # [0, 1.78, 0],
    #
    # [4, 0.98, 1],
    # [4, 1.7, 1],
    # [4, 1.18, 1],
    # [4, 1.34, 1],
    # [4, 1.42, 1],
    # [4, 1.54, 1],
    # [4, 1.78, 1],
    #
    # [4, 0.98, 2],
    # [4, 1.7, 2],
    # [4, 1.18, 2],
    # [4, 1.34, 2],
    # [4, 1.42, 2],
    # [4, 1.54, 2],
    # [4, 1.78, 2],
]

NacCore_coordinates = [
    [1.05, 1, 4],
    [1.1, 0.9, 4.2],
    [0.7, 1, 4.1],
    [0.85, 1, 4.2],
    [0.9, 1, 4.1],
    [0.9, 1.33, 3.9],
    [0.9, 1.41, 3.9],
    [0.85, 1.21, 3.9],
    [0.9, 1.3, 3.9],
    [1.05, 1.3, 4.05],
    [0.85, 1.3, 3.9],
]

mPFC_coordinates = np.array(mPFC_coordinates)
NacCore_coordinates = np.array(NacCore_coordinates)
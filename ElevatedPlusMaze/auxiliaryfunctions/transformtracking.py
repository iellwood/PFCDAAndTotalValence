import numpy as np
import auxiliaryfunctions.EPMGeometry as EPMGeometry
import os
import pickle
def load_dataset_and_rescale_tracking(data_directory_path):
    # data_directory_path = 'FearConditioningData/' + condition + '/'

    file_names = os.listdir(data_directory_path)
    dataset = []
    removed_file_indices = []
    for i, file_name in enumerate(file_names):
        with open(data_directory_path + file_name, "rb") as input_file:
            if file_name != 'FC3_EPM_cleaned.obj':
                dataset.append(pickle.load(input_file))
                print('loaded file,', file_name)
            else:
                print('Removing ' + file_name + ' from NAc cohort as the recording has a giant baseline shift, presumably from a slipped fiber')
                print('This removal does not appear to affect the analysis, qualitatively')
                removed_file_indices.append(i)

    file_names = [file_name for j, file_name in enumerate(file_names) if j not in removed_file_indices]

    def rotate_and_scale_tracking(x, y, coordinates, start_side):
        x[:] = x - coordinates['center'][0]
        y[:] = y - coordinates['center'][1]

        y[:] = y * -1 # This accounts for the upsidedown coordinates of the vertical direction of the video

        e_l = np.array(coordinates['left']) - np.array(coordinates['center'])
        e_r = np.array(coordinates['right']) - np.array(coordinates['center'])
        e_u = np.array(coordinates['up']) - np.array(coordinates['center'])
        e_d = np.array(coordinates['down']) - np.array(coordinates['center'])

        average_distance = 0.25*(np.linalg.norm(e_l) + np.linalg.norm(e_r) + np.linalg.norm(e_u) + np.linalg.norm(e_d))

        x[:] = x / average_distance
        y[:] = y / average_distance
        scaled_width = coordinates['width'] / average_distance

        # Rotate the coordinates so that every mouse was held in the right arm at the start
        if start_side == 'right':
            pass
        elif start_side == 'left':
            xy = np.concatenate([x[:, None], y[:, None]], axis=1)
            xy = np.matmul(xy, np.transpose([[-1, 0], [0, -1]], [1, 0]))
            x[:] = xy[:, 0]
            y[:] = xy[:, 1]
        elif start_side == 'up':
            xy = np.concatenate([x[:, None], y[:, None]], axis=1)
            xy = np.matmul(xy, np.transpose([[0, 1], [-1, 0]], [1, 0]))
            x[:] = xy[:, 0]
            y[:] = xy[:, 1]
        elif start_side == 'down':
            xy = np.concatenate([x[:, None], y[:, None]], axis=1)
            xy = np.matmul(xy, np.transpose([[0, -1], [1, 0]]))
            x[:] = xy[:, 0]
            y[:] = xy[:, 1]
        else:
            raise Exception('Unknown start side: ' + str(start_side))

        return scaled_width

    start_dir = []
    widths = []
    for i, d in enumerate(dataset):
        coordinates = EPMGeometry.EPM_coordinate_dict[file_names[i]]
        start_side = EPMGeometry.EPM_start_side_dict[file_names[i]]
        widths.append(rotate_and_scale_tracking(d['x_coordinate'], d['y_coordinate'], coordinates, start_side))

    average_width = np.mean(widths)
    center = [0, 0]
    fraction_of_radius_in_center = 0.092/2.0
    arm_length = 1.0
    center_radius = average_width/2.0

    return dataset, average_width, center, fraction_of_radius_in_center, arm_length, center_radius











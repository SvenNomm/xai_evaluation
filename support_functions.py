# this files contains support functions

import numpy as np
import itertools


def project_data(data_set, feature_set, idx_column):
    n, m = data_set.shape

    data_set_proj = data_set[:, feature_set]
    if idx_column:
        data_test = data_set[:, m-1].reshape(-1, 1)
        data_set_proj = np.hstack((data_set_proj, data_set[:, m-1].reshape(-1, 1)))

    return data_set_proj


def select_classes(data_set, cluster_set):
    n, m = data_set.shape
    desired_classes = np.array(cluster_set)
    mask = np.isin(element=data_set[:,m-1], test_elements=desired_classes)
    new_data_set = data_set[mask]
    return new_data_set


def get_cube_neighbourhood(point, *delta):
    if not delta:
        delta = 1

    index_list = []
    for coordinate in point:
        temp_list = []
        for j in range(0, 2 * delta + 1):
            temp_list.append(coordinate - delta + j)

        index_list.append(temp_list)

    cube_neighbourhood = list(item for item in itertools.product(*index_list, repeat=1))
    return cube_neighbourhood


def n2one_position_converter(coordinates, grid_shape):
    dimensionality = len(coordinates)
    k = coordinates[0]
    grid_shape = np.array(grid_shape)
    for i in range(1, dimensionality):
        a = np.prod(grid_shape[1:i+1])
        k = k + np.prod(grid_shape[1:i+1]) * (coordinates[i])

    return k


def get_grid(data_set, step_size, idx_column):
    n, m = data_set.shape

    if idx_column:
        m = m-1

    lower_bounds = np.min(data_set[:, 0:m], axis=0)
    upper_bounds = np.max(data_set[:, 0:m], axis=0)
    min_low = np.min(lower_bounds)
    max_upper = np.max(upper_bounds)
    temp_array = np.arange(start=min_low, stop=max_upper, step=step_size)
    #n_temp, _ = temp_array.shape
    n_temp = len(temp_array)

    grid_sources = np.zeros((m, n_temp))
    for j in range(0, m):
        grid_sources[j, :] = temp_array

    grid_elements = np.meshgrid(*grid_sources)
    grid_shape = np.array(grid_elements[0].shape)
    dim_length = np.prod(grid_shape)
    flat_grid = np.zeros((dim_length, m))
    for j in range(0, m):
        flat_grid[:, j] = np.ravel(grid_elements[j])

    return flat_grid, grid_elements


def get_decision_boundary(grid_elements, y_hat, flat_grid):
    m = len(grid_elements)
    internal_shape = grid_elements[0].shape

    temp_idx = np.arange(start=1, stop=internal_shape[0] - 1, step=1)
    idx_array = np.zeros((m, internal_shape[0]))

    index_scales = []
    for j in range(m):
        index_scales.append(temp_idx)

    grid_points = list(item for item in itertools.product(*index_scales, repeat=1))
    decision_boundary_points = []

    for point in grid_points:
        cube_neighbourhood = get_cube_neighbourhood(point)
        internal_shapes = []
        for cnp in cube_neighbourhood:
            internal_shapes.append(internal_shape)
        cube_neighbourhood_k_flattened_1 = map(n2one_position_converter, cube_neighbourhood, internal_shapes)
        cube_neighbourhood_k_flattened_2 = list(cube_neighbourhood_k_flattened_1)
        cube_neighbourhood_k_flattened = list(map(lambda x: int(x), cube_neighbourhood_k_flattened_2))

        neighbourhood_labels = y_hat[cube_neighbourhood_k_flattened]
        if not all(x == neighbourhood_labels[0] for x in neighbourhood_labels):
            k = n2one_position_converter(point, internal_shape)
            decision_boundary_points.append(flat_grid[k, :])

    decision_boundary = np.array(decision_boundary_points)
    return decision_boundary

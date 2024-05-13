# this files contains support functions

import numpy as np
import itertools
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import DistanceMetric

from aix360.metrics.local_metrics import faithfulness_metric
from aix360.metrics.local_metrics import monotonicity_metric

def project_data(data_set, feature_set, idx_column):
    n, m = data_set.shape

    data_set_proj = data_set[:, feature_set]
    if idx_column:
        data_test = data_set[:, m - 1].reshape(-1, 1)
        data_set_proj = np.hstack((data_set_proj, data_set[:, m - 1].reshape(-1, 1)))

    return data_set_proj


def select_classes(data_set, cluster_set):
    n, m = data_set.shape
    desired_classes = np.array(cluster_set)
    mask = np.isin(element=data_set[:, m - 1], test_elements=desired_classes)
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
        a = np.prod(grid_shape[1:i + 1])
        k = k + np.prod(grid_shape[1:i + 1]) * (coordinates[i])

    return k


def get_grid(data_set, step_size, idx_column):
    n, m = data_set.shape

    if idx_column:
        m = m - 1

    lower_bounds = np.min(data_set[:, 0:m], axis=0)
    upper_bounds = np.max(data_set[:, 0:m], axis=0)
    min_low = np.min(lower_bounds)
    max_upper = np.max(upper_bounds)
    temp_array = np.arange(start=min_low, stop=max_upper, step=step_size)
    # n_temp, _ = temp_array.shape
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


def get_epsilon_decision_boundary(grid_elements, y_hat, flat_grid, epsilon, metric_name):
    m = len(grid_elements)
    internal_shape = grid_elements[0].shape

    temp_idx = np.arange(start=1, stop=internal_shape[0] - 1, step=1)
    idx_array = np.zeros((m, internal_shape[0]))

    index_scales = []
    for j in range(m):
        index_scales.append(temp_idx)

    grid_points = list(item for item in itertools.product(*index_scales, repeat=1))
    decision_boundary_points = []

    dist = DistanceMetric.get_metric(metric_name)

    for point in grid_points:
        cube_neighbourhood = get_cube_neighbourhood(point)
        pnt = np.asarray(point).reshape(1, -1)
        internal_shapes = []
        epsilon_neighbourhood = []
        for cnp in cube_neighbourhood:
            internal_shapes.append(internal_shape)
            cnpp = np.asarray(cnp).reshape(1, -1)
            a = dist.pairwise(pnt, cnpp)
            if dist.pairwise(pnt, cnpp)[0, 0] <= epsilon:
                epsilon_neighbourhood.append(cnp)

        cube_neighbourhood_k_flattened_1 = map(n2one_position_converter, cube_neighbourhood, internal_shapes)
        cube_neighbourhood_k_flattened_2 = list(cube_neighbourhood_k_flattened_1)
        cube_neighbourhood_k_flattened = list(map(lambda x: int(x), cube_neighbourhood_k_flattened_2))

        neighbourhood_labels = y_hat[cube_neighbourhood_k_flattened]
        if not all(x == neighbourhood_labels[0] for x in neighbourhood_labels):
            k = n2one_position_converter(point, internal_shape)
            decision_boundary_points.append(flat_grid[k, :])

    decision_boundary = np.array(decision_boundary_points)
    return decision_boundary


def classify_and_explain(X_train, X_test, y_train, y_test, flat_grid, classifier, explainer, data_name, print_metrics):
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_test)
    classifier_name = type(classifier)

    if print_metrics:
        print(data_name, ' ', classifier_name, ' accuracy score: ', accuracy_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' precision score: ', precision_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' recall score: ', recall_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' f1 score: ', f1_score(y_test, y_hat))

    hat_grid = classifier.predict(flat_grid)

    explainer_type_internal = type(explainer).__name__
    print('Using: ', explainer_type_internal)
    exp_grid_weights = []
    if explainer_type_internal == 'LimeTabularExplainer':
        print('lime')
        x = flat_grid[:, 0]
        n_grid = len(flat_grid)
        exp_grid_weights = np.zeros((n_grid, 2))

        for i in range(0, n_grid):
            exp_svc_rot = explainer.explain_instance(data_row=flat_grid[i, :], predict_fn=classifier.predict_proba)
            exp_grid_weights[i, 1] = exp_svc_rot.as_list()[0][1]
            exp_grid_weights[i, 0] = exp_svc_rot.as_list()[1][1]  # NB!  observe the chane of indexes due to the fact
            # that second feature comes first

    elif explainer_type_internal == 'str':
        print('shap')
        explainer_shap = shap.KernelExplainer(classifier.predict_proba, X_train)
        exp_grid_weights = explainer_shap.shap_values(flat_grid)

    return y_hat, hat_grid, exp_grid_weights


def classify_explain_evaluate(X_train, X_test, y_train, y_test, flat_grid, classifier, explainer, data_name,
                              print_metrics):
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_test)
    classifier_name = type(classifier)

    if print_metrics:
        print(data_name, ' ', classifier_name, ' accuracy score: ', accuracy_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' precision score: ', precision_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' recall score: ', recall_score(y_test, y_hat))
        print(data_name, ' ', classifier_name, ' f1 score: ', f1_score(y_test, y_hat))

    hat_grid = classifier.predict(flat_grid)

    explainer_type_internal = type(explainer).__name__
    print('Using: ', explainer_type_internal)
    exp_grid_weights = []
    if explainer_type_internal == 'LimeTabularExplainer':
        print('lime')
        x = flat_grid[:, 0]
        n_grid = len(flat_grid)
        exp_grid_weights = np.zeros((n_grid, 2))
        faithfulness = np.zeros((n_grid, 1))
        monotonicity = np.zeros((n_grid, 1))

        for i in range(0, n_grid):
            exp_svc_rot = explainer.explain_instance(data_row=flat_grid[i, :], predict_fn=classifier.predict_proba)
            le = exp_svc_rot.local_exp[y_hat[i]]
            m =  le.as_map()
            xr = X_test[i, :]
            coefs = np.zeros(xr.shape[0])
            for v in le:
                coefs[v[0]] = v[1]

            base = np.zeros(xr.shape[0])
            faithfulness[i, 0] = faithfulness_metric(classifier, xr, coefs, base)
            monotonicity[i, 0] = monotonicity_metric(classifier, xr, coefs, base)



            exp_grid_weights[i, 1] = exp_svc_rot.as_list()[0][1]
            exp_grid_weights[i, 0] = exp_svc_rot.as_list()[1][1]  # NB!  observe the chane of indexes due to the fact
            # that second feature comes first

    elif explainer_type_internal == 'str':
        print('shap')
        explainer_shap = shap.KernelExplainer(classifier.predict_proba, X_train)
        exp_grid_weights = explainer_shap.shap_values(flat_grid)

    return y_hat, hat_grid, exp_grid_weights, faithfulness, monotonicity
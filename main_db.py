import pickle as pkl
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
from sklearn import svm
from sklearn.model_selection import train_test_split
import support_functions as sf
import sklearn.metrics as skm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import tree
from sklearn.metrics import DistanceMetric
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import itertools
from mpl_toolkits import mplot3d


step_size = 0.2

os_name = os.name
path_win = 'G:/'
path_mac = '/Users/svennomm/Library/CloudStorage/GoogleDrive-sven.nomm@gmail.com/'
folder_1 = 'My Drive/Teaching/Machine_Learning_2024/practice_06/data/'
folder_2 = 'My Drive/Teaching/Machine_Learning_2024/practice_04_supervised_learning_1/data/'

if os_name == 'posix':
    path = path_mac
elif os_name == 'nt':
    path = path_win
else:
    path = '/content/gdrive/'  # assumes that Google Drive is mounted in colab drive.mount('/content/gdrive') ?os.chdir('gdrive/MyDrive/Colab Notebooks')?

#fname = path + folder_1 + 'half_moons.pkl'
fname = path + folder_2 + 'data_set_1_3D_labeled.pkl' # this is three gaussian data set
file_handle = open(fname, 'rb')
data_set = pkl.load(file_handle)
n, m = data_set.shape  # assume that the last column contains labeling information

# eliminate certain classes if needed
data_set = sf.select_classes(data_set, [0, 1])

# plotting is possible for 2 and 3 D cases only, for higher dimensions use projections
if m == 3:
    fig_1, ax_1 = plt.subplots()
    ax_1.scatter(x=data_set[:, 0], y= data_set[:, 1], c=data_set[:, 2], cmap='Paired')
    plt.show()

elif m == 4:
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(projection='3d')
    ax_1.scatter(data_set[:, 0], data_set[:, 1], data_set[:, 2], s=10, c=data_set[:, 3], cmap='Paired')
    plt.show()


data_set_train, data_set_test = train_test_split(data_set, train_size=0.7, test_size=0.3)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_set_train[:, 0:m-1], data_set_train[:, m-1])
y_hat = clf.predict(data_set_test[:, 0:m-1])
print('accuracy: ', skm.accuracy_score(data_set_test[:, m-1], y_hat))
print('precision: ', skm.precision_score(data_set_test[:, m-1], y_hat, average='macro')) # average is needed for multiclass only
print('recall: ', skm.recall_score(data_set_test[:, m-1], y_hat, average='macro'))
print('f1: ', skm.f1_score(data_set_test[:, m-1], y_hat, average='macro'))


flat_grid, grid_elements = sf.get_grid(data_set_train, step_size=step_size, idx_column=True)
y_hat = clf.predict(flat_grid)

decision_boundary = sf.get_decision_boundary(grid_elements, y_hat, flat_grid)

x = flat_grid[:, 0]
y = flat_grid[:, 1]

if m == 3:
    fig_2, ax_2 = plt.subplots()
    ax_2.scatter(flat_grid[:, 0], flat_grid[:, 1], c=y_hat, cmap='Paired', s=1, alpha=0.3)
    ax_2.scatter(data_set_test[:, 0], data_set_test[:, 1], c=data_set_test[:, 2], cmap='Paired', s=10)
    ax_2.scatter(decision_boundary[:, 0], decision_boundary[:, 1], c='red', s=10)
    plt.show()

if m == 4:
    z = flat_grid[:, 2]
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(projection='3d')
    ax_2.scatter(flat_grid[:, 0], flat_grid[:, 1], flat_grid[:, 2], c=y_hat, cmap='Paired', s=1, alpha=0.3)
    ax_2.scatter(data_set_test[:, 0], data_set_test[:, 1], data_set_test[:, 2], s=10, c=data_set_test[:, 3],
                 cmap='tab10')
    ax_2.scatter(decision_boundary[:, 0], decision_boundary[:, 1],  decision_boundary[:, 2], c='red', cmap='Paired',s=10)
    plt.show()


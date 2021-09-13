import pydotplus
import pandas as pd
import numpy as np
from io import StringIO
# from sklearn.externals.six import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from classes import BalancedUndersamplingShuffle, balanced_sampling
from helpers import plot_learning_curve, plot_validation_curve, plot_validation_curve_with_undersampling, \
    fit_and_score_pipeline, fit_and_score_iteratively
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score, make_scorer
from scipy.io import arff

if __name__ == '__main__':
    from scipy.io import arff

    with open('1year.arff', 'r') as f:
        data1, meta1 = arff.loadarff(f)
        data1 = np.asarray(data1.tolist(), dtype=np.float32)
        print(data1.shape)

    with open('2year.arff', 'r') as f:
        data2, meta2 = arff.loadarff(f)
        data2 = np.asarray(data2.tolist(), dtype=np.float32)
        print(data2.shape)

    with open('3year.arff', 'r') as f:
        data3, meta3 = arff.loadarff(f)
        data3 = np.asarray(data3.tolist(), dtype=np.float32)
        print(data3.shape)

    with open('4year.arff', 'r') as f:
        data4, meta4 = arff.loadarff(f)
        data4 = np.asarray(data4.tolist(), dtype=np.float32)
        print(data4.shape)

    with open('5year.arff', 'r') as f:
        data5, meta5 = arff.loadarff(f)
        data5 = np.asarray(data5.tolist(), dtype=np.float32)
        print(data5.shape)
    data = np.concatenate([data1, data2, data3, data4, data5], axis=0)
    data[np.isnan(data)] = 0

    minority_class_shape = data[data[:, -1] == 1].shape
    print(f'label=1 shape = {minority_class_shape}')
    print(f'label=1 pct = {minority_class_shape[0] / data.shape[0]}')

    X, y = data[:, :-1], data[:, -1]

    print(f'X.shape={X.shape}, y.shape={y.shape}')
    X_,  y_, idx = balanced_sampling(X, y, 1, random_state=0)
    # optimize hidden_layer_sizes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # No undersampling
    classifier = SVC()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), classifier)
    train_results, test_results = \
        fit_and_score_iteratively(pipe, X, y, 1, iterations=2, use_validation_set=False,
                                  include_train_results=True)
    print(train_results, test_results)


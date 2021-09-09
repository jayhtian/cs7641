import pydotplus
import pandas as pd
import numpy as np
from io import StringIO
# from sklearn.externals.six import StringIO
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from classes import BalancedUndersamplingShuffle, balanced_sampling
from helpers import plot_learning_curve, plot_validation_curve, plot_validation_curve_with_undersampling, \
    fit_and_score_pipeline
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

    # optimize hidden_layer_sizes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_range = range(1, 4)
    scoring = ['accuracy', 'f1', 'recall', 'precision']

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    classifier = MLPClassifier(solver='sgd', alpha=1e-2, random_state=0, max_iter=1000)

    results_store, plt = plot_validation_curve_with_undersampling(classifier, X_train, y_train,
                                                                  param_name='hidden_layer_sizes',
                                                                  param_range=param_range, fit_params=None,
                                                                  error_score='raise',
                                                                  cv=cv, scoring=scoring, n_jobs=8, iterations=1,
                                                                  undersampling_ratio=1, is_pipe=True, plot_type='log')


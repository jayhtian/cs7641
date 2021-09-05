import pydotplus
import pandas as pd
import numpy as np
from io import StringIO
# from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from classes import BalancedUndersamplingShuffle, balanced_sampling
from helpers import plot_learning_curve, plot_validation_curve
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import ShuffleSplit

from scipy.io import arff

if __name__ == '__main__':

    with open('1year.arff','r') as f:
        data1, meta1 = arff.loadarff(f)
        data1 = np.asarray(data1.tolist(), dtype=np.float32)
        print(data1.shape)

    with open('2year.arff','r') as f:
        data2, meta2 = arff.loadarff(f)
        data2 = np.asarray(data2.tolist(), dtype=np.float32)
        print(data2.shape)

    with open('3year.arff','r') as f:
        data3, meta3 = arff.loadarff(f)
        data3 = np.asarray(data3.tolist(), dtype=np.float32)
        print(data3.shape)

    with open('4year.arff','r') as f:
        data4, meta4 = arff.loadarff(f)
        data4 = np.asarray(data4.tolist(), dtype=np.float32)
        print(data4.shape)

    with open('5year.arff','r') as f:
        data5, meta5 = arff.loadarff(f)
        data5 = np.asarray(data5.tolist(), dtype=np.float32)
        print(data5.shape)


    data = np.concatenate([data1, data2, data3, data4, data5], axis=0)
    data[np.isnan(data)] = 0

    print(f'data.shape={data.shape}')

    X, y = data[:,:-1], data[:, -1]

    print(f'X.shape={X.shape}, y.shape={y.shape}')

    param_range = range(1, 20)
    classifier = DecisionTreeClassifier(random_state=0, criterion='gini')
    cv = BalancedUndersamplingShuffle(n_splits=10, test_size=0.2, random_state=0)
    cv._iter_indices(X, y)

    X_, y_, i = balanced_sampling(X, y)
    np.sum(y_)
    # splits = list(cv.split(X, y))
    # y[splits[7][1]]
    train_scores, test_scores = plot_validation_curve(classifier, X, y,
                      param_name='max_depth', param_range=param_range,
                      cv=cv, scoring='accuracy', n_jobs=1)


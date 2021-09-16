import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from helpers import plot_validation_curve, fit_and_score_iteratively

if __name__ == '__main__':

    with open('mnist_X_train', 'rb') as f1, \
        open('mnist_X_test', 'rb') as f2,\
        open('mnist_y_train', 'rb') as f3,\
        open('mnist_y_test', 'rb') as f4, \
        open('mnist_y_train_ohe', 'rb') as f5, \
        open('mnist_y_test_ohe', 'rb') as f6:
        X_train = np.load(f1)[:10000]
        X_test = np.load(f2)
        y_train = np.load(f3)[:10000]
        y_test = np.load(f4)
        y_train_ohe = np.load(f5)[:10000]
        y_test_ohe = np.load(f6)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, y_train_ohe.shape, y_test_ohe.shape)

    max_depth_range = list(range(1, 11)) + list(range(11, 25, 2))
    max_depth_range.sort()

    # max_depth - accuracy
    # no undersampling
    # Baseline model

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    classifier = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=30, random_state=0, max_iter=1000)
    train_res, test_res = fit_and_score_iteratively(classifier, X_train=X_train, y_train=y_train, X_test=X_test,
                                                    y_test=y_test,
                                                    binary_classification=False, include_train_results=True)
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from helpers import plot_validation_curve

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
    classifier = DecisionTreeClassifier(random_state=0, criterion='gini')
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    train_scores, test_scores = plot_validation_curve(classifier, X_train, y_train,
                          param_name='max_depth', param_range=max_depth_range,
                          cv=cv, scoring='accuracy', n_jobs=8, title='Decision Tree Max Depth')


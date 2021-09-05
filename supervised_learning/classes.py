import numpy as np
from sklearn.model_selection._split import BaseShuffleSplit, _validate_shuffle_split
from sklearn.utils import _deprecate_positional_args, check_random_state


def balanced_sampling_num_samples(y):
    classes = np.unique(y)
    memo = {}
    for c in classes:
        memo[c] = np.argwhere(y == c).flatten()
    smallest_class_size = min([v.shape[0] for k, v in memo.items()])
    return smallest_class_size * len(classes)


def balanced_sampling(X, y, r, random_state):
    """
    Parameters
    ----------
    X
    y
    r: target largest class to smallest class ratio
    random_state

    Returns
    -------

    """
    classes = np.unique(y)
    memo = {}
    for c in classes:
        data = np.argwhere(y == c).flatten()
        memo[c] = {'data': data, 'size': data.shape[0]}
    smallest_class_size = min([v['size'] for k, v in memo.items()])
    largest_class_size = max([v['size'] for k, v in memo.items()])
    gamma = largest_class_size / smallest_class_size

    """
    (1 - a) * 1 + a * gamma = r
    1 - a + a * gamma = r
    a * (gamma - 1) = r - 1
    a = (r - 1) / (gamma - 1)
    """
    a = (r - 1) / (gamma - 1)
    print(f'a={a}, r={r}, gamma={gamma}')

    idx = []
    for k, v in memo.items():
        idx.append(np.random.choice(v['data'], int((1-a) * smallest_class_size) + int(a * v['size']), replace=False))

    rng = check_random_state(random_state)
    idx = rng.permutation(np.concatenate(idx))
    return X[idx, :], y[idx], idx


class BalancedUndersamplingShuffle(BaseShuffleSplit):
    @_deprecate_positional_args
    def __init__(self, n_splits=10, *, test_size=None, train_size=None,
                 random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, groups=None):
        # get all indices of all classes
        n_samples = balanced_sampling_num_samples(y)

        n_train, n_test = BalancedUndersamplingShuffle(
            n_samples, self.test_size, self.train_size,
            default_test_size=self._default_test_size)

        for i in range(self.n_splits):
            # random partition while undersampling label=0
            X_, y_, idx = balanced_sampling(X, y, self.random_state)

            ind_test = idx[:n_test]
            ind_train = idx[n_test:(n_test + n_train)]
            yield ind_train, ind_test


# if __name__ == '__main__':
#     X = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
#     y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
#     cv = BalancedUndersamplingShuffle(n_splits=10, test_size=0.2, random_state=0)
#     sv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)._iter_indices(X, y)
#     samples = balanced_sampling(X, y, 0)
#     splits = list(cv.split(X, y))
#     print(splits)
#     print('hi')

    # splits = list(cv.split(X, y))
    # y[splits[7][1]]
    # train_scores, test_scores = plot_validation_curve(classifier, X, y,
    #                   param_name='max_depth', param_range=param_range,
    #                   cv=cv, scoring='accuracy', n_jobs=1)


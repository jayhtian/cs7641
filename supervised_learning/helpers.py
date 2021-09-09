from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.model_selection import ShuffleSplit

from classes import balanced_sampling


def validation_curve_with_undersampling(estimator, X, y, param_name, param_range, scoring, n_jobs, cv, iterations,
                                        fit_params, error_score, undersampling_ratio, verbose=0, is_pipe=False):
    """

    Parameters
    ----------
    undersampling_ratio: a number representing ratio of majority to minority class.
    verbose
    fit_params
    error_score
    estimator
    X: Not undersampled X_train
    y: Not undersampled y_train
    param_name
    param_range
    scoring
    n_jobs
    cv
    iterations

    Returns
    -------

    """
    results = {}
    for v in param_range:
        results_store = []
        estimator.set_params(**{param_name: v})
        if is_pipe and type(estimator) != sklearn.pipeline.Pipeline:
            _estimator = make_pipeline(StandardScaler(), estimator)
            print(type(_estimator))
        else:
            _estimator = estimator

        for i in range(0, iterations):
            X_balanced, y_balanced, idx = balanced_sampling(X, y, r=undersampling_ratio, random_state=42)
            results_store.append(cross_validate(_estimator, X_balanced, y_balanced, scoring=scoring,
                                                cv=cv, n_jobs=8, verbose=verbose, fit_params=fit_params,
                                                return_train_score=True))
        if not results_store:
            raise RuntimeError('something went wrong. no results')
        v_results_array = results_store[0]
        for r in results_store[1:]:
            for k, val in r.items():
                v_results_array[k] = np.concatenate([v_results_array[k], val])

        for k, val in v_results_array.items():
            mean = np.mean(val)
            std = np.std(val)
            row = np.array([v, mean, std])
            row = row.reshape(1, row.shape[0])
            if k not in results.keys():
                results[k] = row
            else:
                results[k] = np.concatenate([results[k], row], axis=0)

    return results


def plot_validation_curve_with_undersampling(estimator, X, y, param_name, param_range, scoring, n_jobs, cv, iterations,
                                             fit_params, error_score, undersampling_ratio, verbose=0, is_pipe=False,
                                             x_axis_is_log=True):
    _, axes = plt.subplots(1, len(scoring), figsize=(20, 5))

    results = validation_curve_with_undersampling(estimator=estimator, X=X, y=y, param_name=param_name,
                                                  param_range=param_range, scoring=scoring, n_jobs=n_jobs,
                                                  cv=cv, iterations=iterations, fit_params=fit_params,
                                                  error_score=error_score, undersampling_ratio=undersampling_ratio,
                                                  verbose=verbose, is_pipe=is_pipe)

    for i, score in enumerate(scoring):
        train_results = results[f'train_{score}']
        test_results = results[f'test_{score}']
        v = train_results[:, 0]
        assert len(v) == len(
            param_range), f'v = {v}, len(v)={len(v)}. param_range={param_range}, len={len(param_range)}'
        train_scores_mean = train_results[:, 1]
        train_scores_std = train_results[:, 2]
        test_scores_mean = test_results[:, 1]
        test_scores_std = test_results[:, 2]

        axes[i].set_title(f'{score.capitalize()}')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('score')
        axes[i].set_ylim(0.0, 1.1)
        lw = 2
        if x_axis_is_log:
            axes[i].semilogx(param_range, train_scores_mean, label="Training score",
                             color="darkorange", lw=lw)
        else:
            axes[i].plot(param_range, train_scores_mean, label="Training score",
                             color="darkorange", lw=lw)
        axes[i].fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="darkorange", lw=lw)
        if x_axis_is_log:
            axes[i].semilogx(param_range, test_scores_mean, label="Cross-validation score",
                             color="navy", lw=lw)
        else:
            axes[i].plot(param_range, test_scores_mean, label="Cross-validation score",
                         color="navy", lw=lw)
        axes[i].fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="navy", lw=lw)
        axes[i].legend(loc="best")
        axes[i].grid()

    return results, plt


def plot_validation_curve(estimator, X, y, param_name, param_range, scoring, n_jobs, cv):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring=scoring, n_jobs=n_jobs, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    # plt.plot(param_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    # plt.plot(param_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()

    return train_scores, test_scores


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), iterations=1, scoring=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_scores_mean = []
    train_scores_std = []
    test_scores_mean = []
    test_scores_std = []
    fit_times_mean = []
    fit_times_std = []
    for i in range(0, iterations):
        # 10 times undersample and make 10 learning curves
        print(f'iteration {i + 1}')
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True,
                           scoring=scoring)

        train_scores_mean.append(np.mean(train_scores, axis=1))
        train_scores_std.append(np.std(train_scores, axis=1))
        test_scores_mean.append(np.mean(test_scores, axis=1))
        test_scores_std.append(np.std(test_scores, axis=1))
        fit_times_mean.append(np.mean(fit_times, axis=1))
        fit_times_std.append(np.std(fit_times, axis=1))

    train_scores_mean = np.mean(train_scores_mean, axis=0)
    train_scores_std = np.mean(train_scores_std, axis=0)
    test_scores_mean = np.mean(test_scores_mean, axis=0)
    test_scores_std = np.mean(test_scores_std, axis=0)
    fit_times_mean = np.mean(fit_times_mean, axis=0)
    fit_times_std = np.mean(fit_times_std, axis=0)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def fit_and_score_pipeline(estimator, X, y, cv, scoring):
    pipe = make_pipeline(StandardScaler(), estimator)

    train_ind, test_ind = list(cv.split(X, y))[0]
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    pipe.fit(X_train, y_train)  # apply scaling on training data
    scorer = check_scoring(pipe, scoring=scoring)

    return scorer(pipe, X_train, y_train), scorer(pipe, X_test, y_test), pipe


def exp_range(start, end, increment, exp):
    while start < end:
        yield int(start ** exp)
        start += increment

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


def fit_and_score(classifier, X_train, y_train, X_test, y_test, binary_classification=True):
    classifier.fit(X_train, y_train)

    scorer = check_scoring(classifier, scoring='accuracy')
    test_score_a = scorer(classifier, X_test, y_test)

    scorer = check_scoring(classifier, scoring='accuracy')
    train_score_a = scorer(classifier, X_train, y_train)

    if binary_classification:
        scorer = check_scoring(classifier, scoring='f1')
        test_score_f1 = scorer(classifier, X_test, y_test)

        scorer = check_scoring(classifier, scoring='recall')
        test_score_r = scorer(classifier, X_test, y_test)

        scorer = check_scoring(classifier, scoring='precision')
        test_score_p = scorer(classifier, X_test, y_test)

        scorer = check_scoring(classifier, scoring='balanced_accuracy')
        test_score_ba = scorer(classifier, X_test, y_test)

        scorer = check_scoring(classifier, scoring='f1')
        train_score_f1 = scorer(classifier, X_train, y_train)

        scorer = check_scoring(classifier, scoring='recall')
        train_score_r = scorer(classifier, X_train, y_train)

        scorer = check_scoring(classifier, scoring='precision')
        train_score_p = scorer(classifier, X_train, y_train)

        scorer = check_scoring(classifier, scoring='balanced_accuracy')
        train_score_ba = scorer(classifier, X_train, y_train)

        return np.array((train_score_a, train_score_f1, train_score_p, train_score_r, train_score_ba)), \
               np.array((test_score_a, test_score_f1, test_score_p, test_score_r, test_score_ba))
    else:
        return np.array(train_score_a), \
               np.array(test_score_a)


def fit_and_score_iteratively(classifier, X=None, y=None, undersampling_ratio=None, iterations=1,
                              use_validation_set=False, cv=None,
                              include_train_results=False,
                              X_train=None, y_train=None,
                              X_test=None, y_test=None,
                              binary_classification=True,
                              ohe=False, y_train_ohe=None,
                              y_test_ohe=None):
    test_results = []
    train_results = []

    for i in range(iterations):

        # reduce dataset
        if undersampling_ratio:
            X_, y_, idx = balanced_sampling(X, y, r=undersampling_ratio, random_state=42)
        else:
            X_, y_ = np.copy(X), np.copy(y)
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            ...
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=42, stratify=y_)

        if use_validation_set and cv:
            _train_results = []
            _test_results = []
            for train_ind, test_ind in list(cv.split(X_train, y_train)):
                if ohe:

                    _X_train, _y_train, _X_validation, _y_validation = X_train[train_ind], y_train[train_ind], \
                                                                       X_train[test_ind], y_train_ohe[test_ind]
                else:
                    _X_train, _y_train, _X_validation, _y_validation = X_train[train_ind], y_train[train_ind], \
                                                                       X_train[test_ind], y_train[test_ind]
                _train_scores, _test_scores = fit_and_score(classifier, _X_train, _y_train, _X_validation,
                                                            _y_validation,
                                                            binary_classification=binary_classification)
                _train_results.append(_train_scores)
                _test_results.append(_test_scores)
            iteration_train = np.mean(np.array(_train_results), axis=0)
            iteration_test = np.mean(np.array(_test_results), axis=0)
        else:
            if ohe:
                iteration_train, iteration_test = fit_and_score(classifier, X_train, y_train_ohe, X_test, y_test_ohe,
                                                                binary_classification=binary_classification)
            else:
                iteration_train, iteration_test = fit_and_score(classifier, X_train, y_train, X_test, y_test,
                                                                binary_classification=binary_classification)

        train_results.append(iteration_train)
        test_results.append(iteration_test)

    if include_train_results:
        return np.mean(np.array(train_results), axis=0), np.mean(np.array(test_results), axis=0)
    else:
        return np.mean(np.array(test_results), axis=0)


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
        print(f'param={v}')
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
                                             x_axis_is_log=True, param_range_idx=0):
    _, axes = plt.subplots(1, len(scoring), figsize=(20, 5))

    results = validation_curve_with_undersampling(estimator=estimator, X=X, y=y, param_name=param_name,
                                                  param_range=param_range, scoring=scoring, n_jobs=n_jobs,
                                                  cv=cv, iterations=iterations, fit_params=fit_params,
                                                  error_score=error_score, undersampling_ratio=undersampling_ratio,
                                                  verbose=verbose, is_pipe=is_pipe)
    if hasattr(param_range[0], '__len__'):
        param_range = [p[param_range_idx] for p in param_range]

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
        train_scores_mean = train_scores_mean.astype(float)
        train_scores_std = train_scores_std.astype(float)
        test_scores_mean = test_scores_mean.astype(float)
        test_scores_std = test_scores_std.astype(float)
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


def plot_validation_curve(estimator, X, y, param_name, param_range, scoring, n_jobs, cv,
                          title=None, is_log_axis=True, figsize=None, plot_param_index=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring=scoring, n_jobs=n_jobs, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    try:
        plt = plot_curves(param_range, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                    title, scoring, is_log_axis, figsize, plot_param_index, param_name)
    except:
        ...
    train_scores_mean = train_scores_mean.reshape((train_scores_mean.shape[0], 1))
    test_scores_mean = test_scores_mean.reshape((test_scores_mean.shape[0], 1))
    param_range = np.array(param_range)
    if plot_param_index:
        param_range = param_range[:,plot_param_index].reshape(param_range.shape[0], 1)
    else:
        param_range = param_range.reshape(param_range.shape[0], 1)
    return train_scores, test_scores_mean, \
           np.concatenate([param_range, train_scores_mean], axis=1), np.concatenate([param_range, test_scores_mean], axis=1)


def plot_curves(param_range, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                title, scoring, is_log_axis=True, figsize=None, plot_param_index=None, param_name=None):
    if figsize:
        plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.xlabel(f'{param_name}')

    plt.ylabel(scoring)
    plt.ylim(0.0, 1.1)
    lw = 2
    if plot_param_index:
        _param_range = [p[plot_param_index] for p in param_range]
    else:
        _param_range = param_range
    if is_log_axis:
        plt.semilogx(_param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
    else:
        plt.plot(_param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(_param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    if is_log_axis:
        plt.semilogx(_param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
    else:
        plt.plot(_param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(_param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()

    return plt


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), iterations=1, scoring=None, shuffle=True,
                        undersampling = False):
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
    score_times_mean = []
    for i in range(0, iterations):
        # 10 times undersample and make 10 learning curves
        print(f'iteration {i + 1}')
        if undersampling:
            X_, y_, idx = balanced_sampling(X, y, r=1, random_state=42)
        else:
            X_, y_ = X, y
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=42, stratify=y_)

        train_sizes, train_scores, test_scores, fit_times, score_times = \
            learning_curve(estimator, X_train, y_train, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True,
                           scoring=scoring,
                           shuffle=shuffle)

        if np.isnan(train_scores).any():
            print('lol')
            continue
        train_scores_mean.append(np.mean(train_scores, axis=1))
        train_scores_std.append(np.std(train_scores, axis=1))
        test_scores_mean.append(np.mean(test_scores, axis=1))
        test_scores_std.append(np.std(test_scores, axis=1))
        fit_times_mean.append(np.mean(fit_times, axis=1))
        fit_times_std.append(np.std(fit_times, axis=1))
        score_times_mean.append(np.mean(score_times, axis=1))

    train_scores_mean = np.mean(train_scores_mean, axis=0, )
    train_scores_std = np.mean(train_scores_std, axis=0)
    test_scores_mean = np.mean(test_scores_mean, axis=0)
    test_scores_std = np.mean(test_scores_std, axis=0)
    fit_times_mean = np.mean(fit_times_mean, axis=0)
    fit_times_std = np.mean(fit_times_std, axis=0,)
    score_times_mean = np.mean(score_times_mean, axis=0)

    results = dict(train_sizes=train_sizes,
                   train_scores_mean=train_scores_mean,
                   train_scores_std=train_scores_std,
                   test_scores_mean=test_scores_mean,
                   test_scores_std=test_scores_std,
                   fit_times_mean=fit_times_mean,
                   fit_times_std=fit_times_std,
                   fit_times_train_size_ratio=sum(fit_times_mean)/sum(train_sizes),
                   fit_times_test_score_ratio=sum(fit_times_mean)/sum(test_scores_mean),
                   score_times_test_size_ratio=sum(score_times_mean)/sum(train_sizes))

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

    return results


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

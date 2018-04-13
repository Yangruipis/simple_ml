# -*- coding:utf-8 -*-


from sklearn.model_selection import train_test_split

from simple_ml.base.base_enum import CrossValidationType
from simple_ml.base.base_error import *


def cross_validation(model, x, y, method=CrossValidationType.holdout, test_size=0.3, cv=5):
    if not isinstance(x, np.ndarray):
        raise FeatureTypeError

    if x.shape[0] != len(y):
        raise SampleNumberMismatchError

    result = np.zeros(cv)
    if method == CrossValidationType.holdout:
        for i in range(cv):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size)
            model.fit(x_train, y_train)
            result[i] = model.score(x_test, y_test)
        return result
    elif method == CrossValidationType.k_folder:
        ids = np.arange(x.shape[0])
        ids_random = np.random.choice(ids, len(ids), False)
        ids_split = [[] for i in range(cv)]
        for idx in ids_random:
            group_num = idx % cv
            ids_split[group_num].append(idx)

        for i, group in enumerate(ids_split):
            x_test, y_test = x[group], y[group]
            train_ids = sum([i for i in ids_split if i != group], [])
            x_train, y_train = x[train_ids], y[train_ids]
            model.fit(x_train, y_train)
            result[i] = model.score(x_test, y_test)
        return result
    else:
        raise CrossValidationType

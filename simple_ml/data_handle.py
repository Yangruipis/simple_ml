# -*- coding:utf-8 -*-


import numpy as np


def train_test_split(x, y, test_size=0.3, seed=None):
    if seed:
        np.random.seed(seed)
    id_list = np.arange(x.shape[0])
    id_train = np.random.choice(id_list, int(len(id_list)*(1-test_size)), replace=False)
    id_test = np.array([i for i in id_list if i not in id_train])
    return x[id_train, :], y[id_train], x[id_test, :], y[id_test]


def transform_y(y):
    if list(np.unique(y)) == [-1, 1]:
        return np.array([0 if i == -1 else i for i in y])
    else:
        return y

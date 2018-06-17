# -*- coding:utf-8 -*-

import simple_ml.classify_data as sc
import simple_ml.data_handle as sd
from simple_ml.regression import MultiRegression
from simple_ml.evaluation import regression_plot
import numpy as np


def iris_example():
    x, y = sc.get_iris()
    y = x[:, 0]
    x = x[:, 1:]
    x_train, y_train, x_test, y_test = sd.train_test_split(x, y)
    reg = MultiRegression()
    reg.fit(x_train, y_train)
    print(reg.beta)
    print(reg.r_square)
    print(reg.score(x_test, y_test))
    reg.regression_plot(x_test, y_test, col_id=1)


def weighted_iris_example():
    x, y = sc.get_iris()
    weight = np.ones(len(y))
    y = x[:, 0]
    x = x[:, 1:]
    x_train, x_test = x[:int(x.shape[0]*0.7)], x[int(x.shape[0]*0.7):]
    y_train, y_test = y[:int(x.shape[0]*0.7)], y[int(x.shape[0]*0.7):]
    w_train, w_test = weight[:int(x.shape[0]*0.7)], weight[int(x.shape[0]*0.7):]
    reg = MultiRegression()
    reg.fit(x_train, y_train, w_train)
    print(reg.score(x_test, y_test, w_test))
    reg.regression_plot(x_test, y_test, weight=w_test, col_id=1)


if __name__ == '__main__':
    iris_example()
    weighted_iris_example()

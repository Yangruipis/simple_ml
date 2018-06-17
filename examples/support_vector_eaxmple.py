# -*- coding:utf-8 -*-

from simple_ml.support_vector import *
import numpy as np
from simple_ml.classify_data import *
import simple_ml.svm as ss
from simple_ml.data_handle import train_test_split


def iris_svm_example():
    x, y = get_iris()
    # x = x[(y == 0) | (y == 1)]
    # y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)
    svm = SVM(c=0.1)
    svm.fit(x_train, y_train)
    print(svm.predict(x_test), y_test)


def iris_svr_example():
    _x, y = get_iris()
    x = _x[:, 1:]
    y = _x[:, 0]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)
    svr = SVR(c=1.0, eps=0.9, kernel=KernelType.gaussian, sigma=2.0)
    svr.fit(x_train, y_train)
    y_predict = svr.predict(x_test)
    print(np.corrcoef(y_predict.ravel(), y_test))
    for i, j in enumerate(y_predict):
        print(j, y_test[i])
    svr.regression_plot(x_test, y_test, column_id=0)


def multi_svm_example():
    x, y = get_wine()  # get moon()
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)
    svm = SVM(kernel=KernelType.polynomial, d=2)
    svm.fit(x_train, y_train)
    y_predict = svm.predict(x_test)
    print(y_predict, list(y_test))


if __name__ == '__main__':
    # iris_svm_example()
    multi_svm_example()
    # iris_svr_example()

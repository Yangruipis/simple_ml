# -*- coding:utf-8 -*-

from simple_ml.svm import *
from simple_ml.classify_data import *
from simple_ml.helper import train_test_split


def iris_example():
    """
    线性可分情况
    :return:
    """
    x, y = get_iris()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test)


def iris_example2():
    """
    线性不可分，软间隔情况
    :return:
    """
    x, y = get_iris()
    x = x[(y == 1) | (y == 2)]
    y = y[(y == 1) | (y == 2)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test)


def moon_example():
    """
    线性不可分，高维可分情况
    :return:
    """
    x, y = get_wine() # get moon()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test, ", Linear")

    # sigma设置的比较小，会过拟合
    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.gaussian, sigma=0.5)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test, ", Gaussian(sigma=0.5)")

    # sigma设置的比较大，会欠拟合
    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.gaussian, sigma=1)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test, ", Gaussian(sigma=1.0)")

    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.laplace, sigma=1)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test, ", Laplace(sigma=1)")

    mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.sigmoid, beta=1, theta=-1)
    mysvm.fit(x_train, y_train)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(x_train))
    mysvm.classify_plot(x_test, y_test, ", Sigmoid(beta=1,theta=1)")


if __name__ == '__main__':
    # iris_example()
    moon_example()

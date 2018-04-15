# -*- coding:utf-8 -*-

from simple_ml.logistic import *
from simple_ml.classify_data import get_iris, get_wine
from simple_ml.helper import train_test_split


def iris_example():
    x, y = get_iris()
    x = x[(y==0)|(y==1)]
    y = y[(y==0)|(y==1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    logistic = BaseLogisticRegression(has_intercept=False)
    logistic.fit(x_train, y_train)
    print(logistic.w)
    logistic.classify_plot(x_test, y_test)
    logistic.auc_plot(x_test, y_test)

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    print(lasso.w)
    lasso.classify_plot(x_test, y_test)

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    print(ridge.w)
    ridge.classify_plot(x_test, y_test)


def wine_example():
    x, y = get_wine()
    x = x[(y==0)|(y==1)]
    y = y[(y==0)|(y==1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    logistic = BaseLogisticRegression(has_intercept=False)
    logistic.fit(x_train, y_train)
    print(logistic.w)
    logistic.classify_plot(x_test, y_test)
    logistic.auc_plot(x_test, y_test)

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    print(lasso.w)
    lasso.classify_plot(x_test, y_test)

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    print(ridge.w)
    ridge.classify_plot(x_test, y_test)

if __name__ == '__main__':
    wine_example()

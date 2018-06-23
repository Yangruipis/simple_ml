# -*- coding:utf-8 -*-

from simple_ml.logistic import *
from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split


def iris_example():
    x, y = get_iris()
    x = x[(y==0)|(y==1)]
    y = y[(y==0)|(y==1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    print(logistic.w)
    logistic.predict(x_test)
    logistic.score(x_test, y_test)
    logistic.classify_plot(x_test, y_test)
    logistic.auc_plot(x_test, y_test)

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    print(lasso.w)
    lasso.predict(x_test)
    lasso.score(x_test, y_test)
    lasso.classify_plot(x_test, y_test)

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    print(ridge.w)
    ridge.predict(x_test)
    ridge.score(x_test, y_test)
    ridge.classify_plot(x_test, y_test)


def wine_example():
    x, y = get_wine()
    x = x[(y==0)|(y==1)]
    y = y[(y==0)|(y==1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.5, 918)

    logistic = LogisticRegression(has_intercept=True)
    logistic.fit(x_train, y_train)
    logistic.score(x_test, y_test)
    print(logistic.w)
    logistic.classify_plot(x_test, y_test)
    logistic.auc_plot(x_test, y_test)

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    print(lasso.w)
    lasso.classify_plot(x_test, y_test)
    lasso.auc_plot(x_test, y_test)

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    print(ridge.w)
    ridge.classify_plot(x_test, y_test)
    ridge.auc_plot(x_test, y_test)


def multi_class_example():
    x, y = get_wine()
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.5, 918)
    logistic = LogisticRegression(has_intercept=True)
    logistic.fit(x_train, y_train)
    print(logistic.predict(x_test))
    logistic.classify_plot(x_test, y_test)

    logistic = Lasso(has_intercept=True)
    logistic.fit(x_train, y_train)
    print(logistic.predict(x_test))
    logistic.classify_plot(x_test, y_test)

    logistic = Ridge(has_intercept=True)
    logistic.fit(x_train, y_train)
    print(logistic.predict(x_test))
    logistic.classify_plot(x_test, y_test)


if __name__ == '__main__':
    iris_example()
    # wine_example()
    # multi_class_example()

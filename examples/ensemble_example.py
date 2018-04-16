# -*- coding:utf-8 -*-

from simple_ml.ensemble import *
from simple_ml.classify_data import *
from simple_ml.helper import train_test_split


def moon_example():
    """
    AdaBoost的例子，以月亮数据集为例
    :return:
    """
    x, y = get_wine()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    # 采用Logistic回归作为子分类器的AdaBoost
    ada = AdaBoost(classifier=ClassifierType.LR)
    ada.fit(x_train, y_train)
    print(ada.score(x_test, y_test))
    ada.classify_plot(x_test, y_test, ", LR")

    # 采用KNN作为子分类器的AdaBoost
    ada = AdaBoost(classifier=ClassifierType.KNN)
    ada.fit(x_train, y_train)
    print(ada.score(x_test, y_test))
    ada.classify_plot(x_test, y_test, ", KNN")

    # 采用CART树为子分类器的AdaBoost
    ada = AdaBoost(classifier=ClassifierType.CART)
    ada.fit(x_train, y_train)
    print(ada.score(x_test, y_test))
    ada.classify_plot(x_test, y_test, ", CART")


def watermelon_example():
    """
    GBDT的例子，以西瓜数据集为例
    GBDT暂时只支持回归操作，不支持分类
    :return:
    """
    x, y = get_watermelon()
    y = x[:, -1]     # y为连续标签
    x = x[:, :-1]    # x为离散标签
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    gbdt = GBDT(learning_rate=1)
    gbdt.fit(x_train, y_train)
    print(gbdt.predict(x_test), y_test)
    print("R square: %.4f" % gbdt.score(x_test, y_test))

    x, y = get_wine()
    y = x[:, -1]  # y为连续标签
    x = x[:, :-1]  # x为离散标签
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    gbdt = GBDT(learning_rate=1)
    gbdt.fit(x_train, y_train)
    print(gbdt.predict(x_test), y_test)
    print("R square: %.4f" % gbdt.score(x_test, y_test))


if __name__ == '__main__':
    moon_example()
    # watermelon_example()
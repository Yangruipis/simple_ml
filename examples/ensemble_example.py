# -*- coding:utf-8 -*-

from simple_ml.ensemble import *
from simple_ml.classify_data import *
from simple_ml.helper import train_test_split


def moon_example():
    x, y = get_moon()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    ada = BaseAdaBoost()
    ada.fit(x_train, y_train)
    print(ada.score(x_test, y_test))
    ada.classify_plot(x_test, y_test)


def watermelon_example():
    x, y = get_watermelon()
    y = x[:, -1]    # y为连续标签
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
    # moon_example()
    watermelon_example()
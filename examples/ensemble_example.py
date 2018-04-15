# -*- coding:utf-8 -*-

from simple_ml.ensemble import *
from simple_ml.classify_data import *
from simple_ml.helper import train_test_split


def iris_example():
    x, y = get_moon()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    ada = BaseAdaBoost()
    ada.fit(x_train, y_train)
    print(ada.score(x_test, y_test))
    ada.classify_plot(x_test, y_test)

if __name__ == '__main__':
    iris_example()
# -*- coding:utf-8 -*-

from simple_ml.classify_data import *
from simple_ml.knn import *
from simple_ml.helper import train_test_split


def wine_example():
    x, y = get_wine()
    # knn可以解决多分类问题
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    knn = KNN()
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))
    knn.classify_plot(x_test, y_test)


if __name__ == '__main__':
    wine_example()
    # moon_example()

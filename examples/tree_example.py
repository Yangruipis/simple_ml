# -*- coding:utf-8 -*-

from simple_ml.classify_data import *
from simple_ml.tree import *
from simple_ml.helper import train_test_split


def ID3_example():
    x, y = get_watermelon()
    x = x[:, :4]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)
    id3 = ID3()
    id3.fit(x_train, y_train)
    print(id3.score(x_test, y_test))


def wine_example():
    x, y = get_wine()

    #x = x[(y == 0) | (y == 1)]
    #y = y[(y == 0) | (y == 1)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    cart = CART()
    cart.fit(x_train, y_train)
    print(cart.score(x_test, y_test))
    cart.classify_plot(x_test, y_test)

    y = x[:, -1]
    x = x[:, :-1]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    cart = CART()
    cart.fit(x_train, y_train)
    print(cart.score(x_test, y_test))


def random_forest_example():
    x, y = get_wine()

    # x = x[(y == 0) | (y == 1)]
    # y = y[(y == 0) | (y == 1)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    rf = RandomForest(4, 50)
    rf.fit(x_train, y_train)
    print(rf.score(x_test, y_test))
    rf.classify_plot(x_test, y_test)


if __name__ == '__main__':
    # ID3_example()
    # wine_example()
    random_forest_example()

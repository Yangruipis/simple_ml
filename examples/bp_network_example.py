# -*- coding:utf-8 -*-

from simple_ml.classify_data import *
from simple_ml.neural_network import *
from simple_ml.data_handle import train_test_split
from simple_ml.base.base_enum import *


def wine_example():
    x, y = get_wine()

    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    nn = NeuralNetwork(alpha=0.5, cost_func=CostFunction.square)
    nn.clear_all()
    nn.add_some_layers(2, 3, active_func=ActiveFunction.relu)
    nn.fit(x_train, y_train)
    print(nn.predict_prob(x_test))
    nn.classify_plot(x_test, y_test)
    nn.auc_plot(x_test, y_test)

def moon_example():
    x, y = get_moon()

    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    nn = NeuralNetwork(alpha=0.5, cost_func=CostFunction.square)
    nn.clear_all()
    nn.add_some_layers(2, 3, active_func=ActiveFunction.relu)
    nn.fit(x_train, y_train)
    print(nn.predict_prob(x_test))
    nn.classify_plot(x_test, y_test)
    nn.auc_plot(x_test, y_test)


def multi_class_example():
    x, y = get_wine()

    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    nn = NeuralNetwork(alpha=0.5, cost_func=CostFunction.square)
    nn.clear_all()
    nn.add_some_layers(2, 3, active_func=ActiveFunction.relu)
    nn.fit(x_train, y_train)
    # print(nn.predict_prob(x_test))    # we raise error here
    nn.classify_plot(x_test, y_test)
    # nn.auc_plot(x_test, y_test)       # we raise error here


if __name__ == '__main__':
    wine_example()
    # moon_example()
    # multi_class_example()
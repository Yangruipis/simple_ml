# -*- coding:utf-8 -*-

import simple_ml.classify_data as sc
import simple_ml.data_handle as sd
from simple_ml.regression import MultiRegression
from simple_ml.evaluation import regression_plot


def iris_example():
    x, y = sc.get_iris()
    y = x[:, 0]
    x = x[:, 1:]
    x_train, y_train, x_test, y_test = sd.train_test_split(x, y)
    reg = MultiRegression()
    reg.fit(x_train, y_train)
    print(reg.beta)
    print(reg.r_square)
    print(reg.score(x_test, y_test))
    y_predict = reg.predict(x_test)
    regression_plot(x_train, y_train, x_test, y_predict, 2)


if __name__ == '__main__':
    iris_example()
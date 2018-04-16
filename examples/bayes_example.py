# -*- coding:utf-8 -*-

from simple_ml.classify_data import *
from simple_ml.bayes import *
from simple_ml.helper import train_test_split


def wine_example():
    x, y = get_wine()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    # 贝叶斯最小错误率
    bme = BaseBayesMinimumError()
    bme.fit(x_train, y_train)
    print(bme.score(x_test, y_test))
    bme.classify_plot(x_test, y_test)

    # 贝叶斯最小风险，需要给定风险矩阵
    # 风险矩阵 [[0,100], [10,0]] 表示把0分为1（存伪）的损失为100，把1分为0（弃真）的损失为10
    bmr = BayesMinimumRisk(np.array([[0, 100], [10, 0]]))
    bmr.fit(x_train, y_train)
    bmr.predict(x_test)
    print(bmr.score(x_test, y_test))
    bmr.classify_plot(x_test, y_test)

    # 朴素贝叶斯
    nb = NaiveBayes()
    nb.fit(x_train, y_train)
    nb.predict(x_test)
    print(nb.score(x_test, y_test))
    nb.classify_plot(x_test, y_test)


def moon_example():
    x, y = get_moon()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

    # 贝叶斯最小错误率
    bme = BaseBayesMinimumError()
    bme.fit(x_train, y_train)
    print(bme.score(x_test, y_test))
    bme.classify_plot(x_test, y_test)

    # 贝叶斯最小风险，需要给定风险矩阵
    # 风险矩阵 [[0,100], [10,0]] 表示把0分为1（存伪）的损失为100，把1分为0（弃真）的损失为10
    bmr = BayesMinimumRisk(np.array([[0, 100], [10, 0]]))
    bmr.fit(x_train, y_train)
    bmr.predict(x_test)
    print(bmr.score(x_test, y_test))
    bmr.classify_plot(x_test, y_test)

    # 朴素贝叶斯
    nb = NaiveBayes()
    nb.fit(x_train, y_train)
    nb.predict(x_test)
    print(nb.score(x_test, y_test))
    nb.classify_plot(x_test, y_test)


if __name__ == '__main__':
    wine_example()
    moon_example()

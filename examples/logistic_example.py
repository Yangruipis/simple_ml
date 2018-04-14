# -*- coding:utf-8 -*-

from simple_ml.logistic import *

if __name__ == '__main__':
    x = np.array([[1, 0],
                  [1, 1],
                  [0, 1],
                  [0, 0]])
    y = np.array([1, 1, 0, 0])
    logistic = BaseLogisticRegression()
    logistic.fit(x, y)
    print(logistic.w)
    logistic.classify_plot(x, y)
    lasso = Lasso()
    lasso.fit(x, y)
    print(lasso.w)
    lasso.classify_plot(x, y)
    ridge = Ridge()
    ridge.fit(x, y)
    print(ridge.w)
    ridge.classify_plot(x, y)
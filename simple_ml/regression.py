# -*- coding:utf-8 -*-

from simple_ml.base.base_model import *
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.evaluation import regression_r2
import numpy as np


class MultiRegression(BaseClassifier):

    def __init__(self, has_intercept=False, weight=None):
        super(MultiRegression, self).__init__()
        self._function = Function.regression
        self.weight = weight
        self.has_intercept = has_intercept
        self._kernel_mat = None                 # (X^T X)^-1 X^T
        self._beta_hat = None
        self._sigma_beta_hat = None
        self._r_square = None

    def fit(self, x, y):
        super(MultiRegression, self).fit(x, y)
        self.y = y
        self._fit()

    def _add_ones(self, mat):
        return np.column_stack((np.arange(mat.shape[0]), mat))

    def _fit(self):
        # 判断矩阵奇异性质 np.linalg.det ， 或者直接try except(错误类型), e:
        if self.has_intercept:
            self.x = self._add_ones(self.x)
            self.variable_num += 1

        temp = np.dot(self.x.T, self.x)
        if np.linalg.det(temp) != 0:
            self._kernel_mat = np.array(np.mat(np.dot(self.x.T, self.x)).I)          # k * k
            self._beta_hat = np.dot(np.dot(self._kernel_mat, self.x.T), self.y).T   # k x 1
            _y_hat = np.dot(self.x, self._beta_hat)                   # n x 1
            _sigma_hat = np.sum(np.square(self.y - _y_hat)) / (self.sample_num - self.variable_num)  # 1x1
            self._sigma_beta_hat = np.dot(_sigma_hat, self._kernel_mat)  # k x k  只看主对角线
            self._r_square = regression_r2(_y_hat.ravel(), self.y)
        else:
            raise ParamInputError("输入的样本矩阵必须为非奇异矩阵！")

    def predict(self, x):
        if self._kernel_mat is None:
            raise ModelNotFittedError
        if self.has_intercept:
            x = self._add_ones(x)
        if x.shape[1] != self.x.shape[1]:
            raise FeatureNumberMismatchError

        return np.dot(x, self._beta_hat)

    @property
    def beta(self):
        return self._beta_hat

    @property
    def sigma(self):
        return self._sigma_beta_hat

    @property
    def r_square(self):
        return self._r_square

    def score(self, x, y):
        y_predict = self.predict(x)
        return regression_r2(y_predict, y)

    def new(self):
        return MultiRegression(self.has_intercept, self.weight)

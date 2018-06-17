# -*- coding:utf-8 -*-

from simple_ml.base.base_model import *
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.evaluation import regression_r2, regression_plot
import numpy as np


class MultiRegression(BaseClassifier):

    def __init__(self, has_intercept=False):
        super(MultiRegression, self).__init__()
        self._function = Function.regression
        self.has_intercept = has_intercept
        self._kernel_mat = None                 # (X^T X)^-1 X^T
        self._beta_hat = None
        self._sigma_beta_hat = None
        self._r_square = None
        self.weight = None

    def fit(self, x, y, weight=None):
        super(MultiRegression, self).fit(x, y)
        self.weight = weight
        self.y = y  # 此时一定是连续型变量，不需要进行转换，为了防止出错（识别成多分类），在此重新赋值
        if self.weight is not None:
            self._check_weight()
        self._fit()

    def _check_weight(self):
        if len(self.weight.shape) != 1:
            raise InputTypeError("输入权重必须为一维数组")
        if self.weight.shape[0] != self.sample_num:
            raise SampleNumberMismatchError("输入权重与训练集样本数不匹配")

    @staticmethod
    def _add_ones(mat):
        return np.column_stack((np.arange(mat.shape[0]), mat))

    def _fit(self):
        # 判断矩阵奇异性质 np.linalg.det ， 或者直接try except(错误类型), e:
        if self.has_intercept:
            self.x = self._add_ones(self.x)
            self.variable_num += 1

        if self.sample_num < self.variable_num:
            raise ParamInputError("样本数少于待估参数数目，自由度不够")

        if self.weight is None:
            temp = np.dot(self.x.T, self.x)
            if np.linalg.det(temp) != 0:
                self._kernel_mat = np.array(np.mat(temp).I)          # k * k
                self._beta_hat = np.dot(np.dot(self._kernel_mat, self.x.T), self.y).T   # k x 1

            else:
                raise ParamInputError("输入的样本矩阵必须为非奇异矩阵！")
        else:
            temp = np.dot(self.x.T, self.weight.reshape(-1, 1) * self.x)
            if np.linalg.det(temp) != 0:
                self._kernel_mat = np.array(np.mat(temp).I)
                self._beta_hat = np.dot(np.dot(self._kernel_mat, self.x.T),
                                        self.weight * self.y)   # k x 1
            else:
                raise ParamInputError("输入的样本矩阵必须为非奇异矩阵！")
        _y_hat = np.dot(self.x, self._beta_hat)  # n x 1
        _sigma_hat = np.sum(np.square(self.y - _y_hat)) / (self.sample_num - self.variable_num)  # 1x1
        self._sigma_beta_hat = np.dot(_sigma_hat, self._kernel_mat)  # k x k  只看主对角线
        self._r_square = regression_r2(_y_hat.ravel(), self.y)

    def predict(self, x, weight=None):
        if self._kernel_mat is None:
            raise ModelNotFittedError
        if self.has_intercept:
            x = self._add_ones(x)
        if x.shape[1] != self.x.shape[1]:
            raise FeatureNumberMismatchError

        if self.weight is not None and weight is None:
            raise EmptyInputError("必须输入权重")
        if weight is not None and x.shape[0] != len(weight):
            raise SampleNumberMismatchError("样本权重和测试样本长度必须一致")

        if weight is not None:
            return np.dot(weight.reshape(-1, 1) * x, self._beta_hat)
        else:
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

    def score(self, x, y, weight=None):
        y_predict = self.predict(x, weight)
        return regression_r2(y_predict, y)

    def new(self):
        return MultiRegression(self.has_intercept)

    def regression_plot(self, x, y, weight=None, col_id=None, title=""):
        if self._beta_hat is None:
            raise ModelNotFittedError
        y_predict = self.predict(x, weight)
        regression_plot(self.x, self.y, x, y, y_predict,x_column_id=col_id, title=title)

# -*- coding:utf-8 -*-

import numpy as np
from openopt import QP
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.evaluation import *


__all__ = [
    'SVR',
    'SVM',
]


# 大于该值则为支持向量
MIN_SUPPORT_VECTOR_THRESHOLD = 1e-5


class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """

    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def polykernel(dimension, offset=0.0):
        return lambda x, y: (offset + np.inner(x, y)) ** dimension

    @classmethod
    def inhomogenous_polynomial(cls, dimension):
        return cls.polykernel(dimension=dimension, offset=1.0)

    @classmethod
    def homogenous_polynomial(cls, dimension):
        return cls.polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        return lambda x, y: np.tanh(kappa * np.dot(x, y) + c)

    @staticmethod
    def radial_basis(gamma=10):
        return lambda x, y: np.exp(-gamma * np.linalg.norm(np.subtract(x, y)))

    @staticmethod
    def laplace(sigma):
        return lambda x, y: np.exp(- np.sqrt(np.sum((x - y) ** 2)) / sigma)


class BaseSupportVector:

    @staticmethod
    def _get_kernel_func(kernel_name, kwargs):
        if kernel_name == KernelType.linear:
            return Kernel.linear()
        elif kernel_name == KernelType.gaussian:
            return Kernel.gaussian(kwargs['sigma'])
        elif kernel_name == KernelType.polynomial:
            if 'o' in kwargs:
                return Kernel.polykernel(kwargs['d'], kwargs['o'])
            else:
                return Kernel.polykernel(kwargs['d'])
        elif kernel_name == KernelType.laplace:
            return Kernel.laplace(kwargs['sigma'])
        else:
            raise KernelTypeError("非法的核函数名称")


class SVR(BaseSupportVector, BaseClassifier):

    def __init__(self, c, eps=0.1, kernel=KernelType.linear, **kwargs):
        super(SVR, self).__init__()
        self._kernel_name = kernel
        self._kernel = self._get_kernel_func(kernel, kwargs)
        self.C = c
        self._kwargs = kwargs
        self.eps = eps
        self._w = None
        self._b = None

    def fit(self, x, y):
        super(SVR, self).fit(x, y)
        self._fit()

    def _fit(self):
        # 列出openopt所需的约束条件的标准型
        # QP: constructor for Quadratic Problem assignment
        # 1 / 2 x' H x  + f'x -> min
        # subjected to Ax <= b
        # Aeq x = beq
        # lb <= x <= ub
        kernel = np.zeros((self.sample_num * 2, self.sample_num * 2))
        for i in range(self.sample_num):
            for j in range(self.sample_num):
                kernel[i][j] = self._kernel(self.x[i], self.x[j])
                kernel[self.sample_num + i][self.sample_num + j] = kernel[i][j]
                kernel[i + self.sample_num][j] = -1.0 * kernel[i][j]
                kernel[i][self.sample_num + j] = -1.0 * kernel[i][j]

        f = np.zeros(self.sample_num * 2)
        for i in range(self.sample_num):
            f[i] = 1.0 * self.y[i] + self.eps
            f[i + self.sample_num] = -1.0 * self.y[i] + self.eps

        lower_bound = np.zeros(self.sample_num * 2)
        upper_bound = np.ones(self.sample_num * 2) * self.C
        aeq = np.append(np.ones(self.sample_num), -1.0*np.ones(self.sample_num))
        beq = 0.0

        eq = QP(np.asmatrix(kernel), np.asmatrix(f), lb=np.asmatrix(lower_bound),
                        ub=np.asmatrix(upper_bound), Aeq=aeq, beq=beq)
        p = eq._solve('cvxopt_qp', iprint=0)
        f_optimized, alpha = p.ff, p.xf

        # 注意这里的_w 是向量：alpha_i - alpha^*_i，
        # 因为在不同核下需要计算内积，不能写成论文中 \sum_i (alpha_i - alpha^*_i)x_i的形式
        self._w = alpha[:self.sample_num] - alpha[self.sample_num:]  # 1 x n
        b_array = self.y - self._inner_mat(self.x) - self.eps
        self._b = np.mean(b_array)

    def _inner_array(self, x):
        """
        求内积 <w, x> 即 \sum_i (alpha_i - alpha^*_i) <x_i, x>
        :param x:  array输入向量，不是矩阵
        :return: float
        """
        res = 0.0
        for i in range(self.sample_num):
            res += self._w[i] * self._kernel(self.x[i], x)
        return res

    def _inner_mat(self, x):
        """
        同样求内积
        :param x: 样本矩阵
        :return: array
        """
        return np.array([self._inner_array(i) for i in x])

    @property
    def weight(self):
        return self._w

    @property
    def bias(self):
        return self._b

    @property
    def support_vector_id(self):
        return np.arange(self.sample_num)[self._w > MIN_SUPPORT_VECTOR_THRESHOLD]

    def predict(self, x):
        if self._w is None:
            raise ModelNotFittedError

        return self._inner_mat(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        return regression_r2(y_predict, y)

    def regression_plot(self, x):
        y = self.predict(x)
        regression_plot(self.x, self.y, x, y, title="SVR")

    def new(self):
        return SVR(self.C, self.eps, self._kernel_name, **self._kwargs)


class SVM(BaseClassifier, BaseSupportVector):

    def __init__(self, c, eps=0.1, kernel=KernelType.linear, **kwargs):
        super(SVM, self).__init__()
        self._kernel_name = kernel
        self._kernel = self._get_kernel_func(kernel, kwargs)
        self._kwargs = kwargs
        self.C = c
        self.eps = eps
        self._w = None
        self._b = None

    def fit(self, x, y):
        super(SVM, self).fit(x, y)
        self._fit()

    def _fit(self):
        # 列出openopt所需的约束条件的标准型
        # QP: constructor for Quadratic Problem assignment
        # 1 / 2 x' H x  + f'x -> min
        # subjected to Ax <= b
        # Aeq x = beq
        # lb <= x <= ub
        kernel = np.zeros((self.sample_num, self.sample_num))
        for i in range(self.sample_num):
            for j in range(self.sample_num):
                kernel[i, j] = self._kernel(self.x[i], self.x[j])
        kernel = np.multiply(np.outer(self.y, self.y), kernel)
        f = -1.0 * np.ones(self.sample_num)
        A_std = np.diag(np.ones(self.sample_num) * -1.0)
        b_std = np.zeros(self.sample_num)

        A_slack = np.diag(np.ones(self.sample_num))
        b_slack = np.ones(self.sample_num) * self.C

        A = np.vstack((A_std, A_slack))
        b = np.vstack((b_std, b_slack))
        Aeq = self.y
        beq = 0.0

        eq = QP(np.asmatrix(kernel), np.asmatrix(f), A=np.asmatrix(A),
                b=np.asmatrix(b), Aeq=Aeq, beq=beq)
        p = eq._solve('cvxopt_qp', iprint=0)
        f_optimized, alpha = p.ff, p.xf
        self._w = np.multiply(alpha, self.y)
        b_array = self.y - self._inner_mat(self.x)
        self._b = np.mean(b_array)

    def _inner_array(self, x):
        """
        求内积 <w, x> 即 \sum_i (alpha_i - alpha^*_i) <x_i, x>
        :param x:  array输入向量，不是矩阵
        :return: float
        """
        res = 0.0
        for i in range(self.sample_num):
            res += self._w[i] * self._kernel(self.x[i], x)
        return res

    def _inner_mat(self, x):
        """
        同样求内积
        :param x: 样本矩阵
        :return: array
        """
        return np.array([self._inner_array(i) for i in x])

    @property
    def weight(self):
        return self._w

    @property
    def bias(self):
        return self._b

    @property
    def support_vector_id(self):
        return np.arange(self.sample_num)[self._w > MIN_SUPPORT_VECTOR_THRESHOLD]

    def predict(self, x):
        if self._w is None:
            raise ModelNotFittedError

        return np.sign(self._inner_mat(x))

    def score(self, x, y):
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(), self.x, self.y, x, y, title=self.__doc__ + title)

    def new(self):
        return SVR(self.C, self.eps, self._kernel_name, **self._kwargs)

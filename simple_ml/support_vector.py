# -*- coding:utf-8 -*-

import numpy as np
import cvxopt
from simple_ml.base.base_model import *
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.evaluation import *


__all__ = [
    'SVR',
    'SVM',
    'KernelType',
]


# 大于该值则为支持向量
MIN_SUPPORT_VECTOR_THRESHOLD = 1e-5


class Kernel(object):
    """
    Implements list of kernels from
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
        elif kernel_name == KernelType.sigmoid:
            return Kernel.hyperbolic_tangent(kwargs['beta'], kwargs['theta'])
        else:
            raise KernelTypeError("非法的核函数名称")


class SVR(BaseSupportVector, BaseClassifier):

    __doc__ = "Support Vector Regression"

    def __init__(self, c=None, eps=0.1, kernel=KernelType.linear, **kwargs):
        """
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束）
            eps         容忍的带宽
            kernel_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        super(SVR, self).__init__()
        self._function = Function.regression
        self._kernel_name = kernel
        self._kernel = self._get_kernel_func(kernel, kwargs)
        self.C = c
        self._kwargs = kwargs
        self.eps = eps
        self._w = None
        self._b = None

    def fit(self, x, y):
        super(SVR, self).fit(x, y)
        self.y = y  # 此时一定是连续型变量，不需要进行转换，为了防止出错（识别成多分类），在此重新赋值
        self._fit()

    def _fit(self):
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

        P = cvxopt.matrix(kernel)

        q = np.zeros(self.sample_num * 2)
        for i in range(self.sample_num):
            q[i] = 1.0 * self.y[i] + self.eps
            q[i + self.sample_num] = -1.0 * self.y[i] + self.eps
        q = cvxopt.matrix(q)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(self.sample_num * 2) * -1.0))
            h = cvxopt.matrix(np.zeros(self.sample_num * 2))
        else:
            tmp1 = -1.0 * np.diag(np.ones(self.sample_num * 2))
            tmp2 = np.diag(np.ones(self.sample_num * 2))
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.sample_num * 2)
            tmp2 = np.ones(self.sample_num * 2) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        A = cvxopt.matrix(np.append(np.ones(self.sample_num), -1.0*np.ones(self.sample_num)), (1, self.sample_num*2))
        b = cvxopt.matrix(0.0)

        # upper_bound = np.ones(self.sample_num * 2) * self.C

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])

        # 注意这里的_w 是向量：alpha_i - alpha^*_i，
        # 因为在不同核下需要计算内积，不能写成论文中 \sum_i (alpha_i - alpha^*_i)x_i的形式
        self._w = (alpha[:self.sample_num] - alpha[self.sample_num:]).ravel()  # 1 x n
        b_array = self.y - self._inner_mat(self.x)   # + self.eps
        self._b = np.mean(b_array[self.support_vector_id])

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
        return np.arange(self.sample_num)[(self._w > MIN_SUPPORT_VECTOR_THRESHOLD) | (self._w < -MIN_SUPPORT_VECTOR_THRESHOLD)]

    def predict(self, x):
        if self._w is None:
            raise ModelNotFittedError

        return self._inner_mat(x) + self._b

    def score(self, x, y):
        y_predict = self.predict(x)
        return regression_r2(y_predict, y)

    def regression_plot(self, x, y, column_id=0):
        y_predict = self.predict(x)
        regression_plot(self.x, self.y, x, y, y_predict, x_column_id=column_id, title="SVR")

    def new(self):
        return SVR(self.C, self.eps, self._kernel_name, **self._kwargs)


class SVM(BaseClassifier, BaseSupportVector, Multi2Binary):

    __doc__ = "Support Vector Machine"

    def __init__(self, c=None, eps=0.1, kernel=KernelType.linear, **kwargs):
        """
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束），如果为None意味着不添加松弛变量
            eps         容忍的带宽
            kernel_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        super(SVM, self).__init__()
        self._kernel_name = kernel
        self._kernel = self._get_kernel_func(kernel, kwargs)
        self._kwargs = kwargs
        self.C = c
        self.eps = eps
        self._w = None
        self._b = None

    def _adj_y(self, y):
        y_unique = np.unique(y)
        self.y_dic2 = {}
        self.y_dic2[y_unique[0]] = -1
        self.y_dic2[y_unique[1]] = 1
        return np.array([self.y_dic2[i] for i in y])

    def _adj_y_back(self, y):
        dic_rev = {}
        if hasattr(self, 'y_dic2'):
            for i, j in self.y_dic2.items():
                dic_rev[j] = i
            return np.array([dic_rev[i] for i in y])
        return y

    def fit(self, x, y):
        super(SVM, self).fit(x, y)
        if self.label_type == LabelType.binary:

            self.y = self._adj_y(self.y)

            # 事先计算出核函数矩阵，避免高维下的计算问题
            self._fit()
        elif self.label_type == LabelType.multi_class:
            self._multi_fit(self)
        else:
            raise LabelTypeError("SVM不支持连续标签（回归），请使用SVR")

    def _fit(self):
        # 列出cvxopt所需的约束条件的标准型
        # Solves a quadratic program
        #
        #     minimize    (1/2)*x'*P*x + q'*x
        #     subject to  G*x <= h
        #                 A*x = b
        # ref: http://goelhardik.github.io/2016/11/28/svm-cvxopt/
        kernel = np.zeros((self.sample_num, self.sample_num))
        for i in range(self.sample_num):
            for j in range(self.sample_num):
                kernel[i, j] = self._kernel(self.x[i], self.x[j])
        P = cvxopt.matrix(np.multiply(np.outer(self.y, self.y), kernel))
        q = cvxopt.matrix(-1.0 * np.ones(self.sample_num))
        A = cvxopt.matrix(1.0 * self.y, (1, self.sample_num))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(-1.0 * np.eye(self.sample_num))
            h = cvxopt.matrix(np.zeros(self.sample_num))
        else:
            tmp1 = -1.0 * np.eye(self.sample_num)
            tmp2 = np.eye(self.sample_num)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.sample_num)
            tmp2 = np.ones(self.sample_num) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])
        self._w = np.multiply(alpha.ravel(), self.y)
        b_array = self.y - self._inner_mat(self.x)
        # if len(self.support_vector_id) == 0:
        #     raise ValueError
        self._b = np.mean(b_array[self.support_vector_id])

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
        if self._w is None and self.new_models is None:
            raise ModelNotFittedError

        if self.label_type == LabelType.binary:
            return self._adj_y_back(np.sign(self._inner_mat(x) + self._b))  # 别忘了加偏移项！！
        else:
            return self._multi_predict(x)

    def score(self, x, y):
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(), self.x, self.y, x, y, title=self.__doc__ + title)

    def new(self):
        return SVM(self.C, self.eps, self._kernel_name, **self._kwargs)

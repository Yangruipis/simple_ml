# -*- coding:utf-8 -*-

import numpy as np

from simple_ml.base.base_error import *
from simple_ml.base.base import BaseTransform


class PCA(BaseTransform):

    def __init__(self, top_n):
        super(PCA, self).__init__()
        self.top_n = top_n
        self._variable_num = None
        self._eigen_value = None
        self._eigen_vector = None
        self._explain = None
        self._top_n_index = None

    def fit(self, x, y=None):
        self._variable_num = x.shape[1]
        if self.top_n > self._variable_num:
            raise TopNTooLargeError
        self._fit(x)

    def _fit(self, x):
        cov_mat = np.cov(x.T)
        # 下式得到的特征值和特征矩阵有如下特点：
        #     1. 特征矩阵每列表示当前特征值对应的特征向量
        #     2. 特征值没有按照从大到小排列
        self._eigen_value, self._eigen_vector = np.linalg.eig(cov_mat)
        self._top_n_index = self._eigen_value.argsort()[-self.top_n:]

    @property
    def explain_ratio(self):
        return self._explain

    def transform(self, x):
        if x.shape[1] != self._variable_num:
            raise FeatureNumberMismatchError

        if self._eigen_value is None:
            raise ModelNotFittedError
        self._explain = np.sum(self._eigen_value[self._top_n_index]) / np.sum(self._eigen_value)
        new_x = np.array([self._transform_single(i) for i in x])
        return new_x

    def _transform_single(self, row):
        res = [np.dot(row, i) for i in self._eigen_vector.T[self._top_n_index]]
        return np.array(res)

    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)


class SuperPCA(PCA):

    def __init__(self, top_n):
        """
        针对数据维度大于样本数目的情况，可以通过矩阵分解简化计算，但是最多只能得到等同于样本数目的主成分个数：
            Pv = lambda v
            XX'v = lambda v
            X'XX'v = X'lambda v
            sigma x' v = lambda X'v
            sigma (X'v) = lambda (X'v)
        所以只要求 XX'的主成分即可，其中X为去每列减去均值后除以根号（总行数-1），保证协方差矩阵sigma等于 X'X
        :param top_n: 主成分个数，当大于样本个数时报错
        """
        super(SuperPCA, self).__init__(top_n)

    def fit(self, x, y=None):
        _sample_number = x.shape[0]
        self._variable_num = x.shape[1]
        if _sample_number == 1:
            raise NeedMoreSampleError

        if _sample_number > self._variable_num:
            super(SuperPCA, self).fit(x)
        else:
            if self.top_n > _sample_number:
                raise TopNTooLargeError

            self._fit(x)

    def _fit(self, x):
        x_new = x.copy()
        for i in range(x_new.shape[0]):
            x_new[:, i] = (x_new[:, i] - np.mean(x_new[:, i])) * 1.0 / np.sqrt(x.shape[0] - 1)

        p = np.dot(x_new, x_new.T)
        self._eigen_value, _eigen_vector = np.linalg.eig(p)    # 得到n个特征值
        self._eigen_vector = np.dot(x.T, _eigen_vector)    # 得到p行n列的特征矩阵， n<<p
        self._top_n_index = self._eigen_value.argsort()[-self.top_n:]

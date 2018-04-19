# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *


class BaseClassifier(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def _init(self, x, y):
        self._clear()
        x_sample_num = self._check_x(x)
        y_sample_num = self._check_y(y)
        if x_sample_num != y_sample_num:
            raise SampleNumberMismatchError
        self.sample_num = x_sample_num
        self.variable_num = x.shape[1]
        self.x = np.array(x)
        self.y = y
        self.label_type = self._check_label_type(self.y)
        self.feature_type = self._check_feature_type(self.x)

    def _clear(self):
        self.x = None
        self.y = None
        self.sample_num = 0
        self.variable_num = 0

    @staticmethod
    def _check_x(x):
        if isinstance(x, np.ndarray):
            if len(x.shape) != 2:
                raise FeatureTypeError("请输入二维数组")
            return x.shape[0]
        else:
            raise FeatureTypeError("请输入二维Numpy.array或pandas.Series")

    @staticmethod
    def _check_y(y):
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise LabelArrayTypeError("请输入一维数组")
            return y.shape[0]
        else:
            raise LabelArrayTypeError("请输入一维Numpy.array或pandas.Series")

    @staticmethod
    def _check_label_type(y):
        count = np.unique(y)
        if len(count) == 2:
            return LabelType.binary
        elif len(count) > len(y)/2:
            return LabelType.continuous
        else:
            return LabelType.multi_class

    @staticmethod
    def _check_feature_type(x):
        res = []
        for feature in x.T:
            count = np.unique(feature)
            if len(count) == 2:
                res.append(LabelType.binary)
            elif len(count) > len(feature) // 2:
                res.append(LabelType.continuous)
            else:
                res.append(LabelType.multi_class)
        return res

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def score(self, x, y):
        pass


class BaseTransform(object):

    __doc__ = "This is a Transform Abstract Class"

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def _init(self, x, y):
        self._clear()
        x_sample_num = self._check_x(x)
        y_sample_num = self._check_y(y)
        if x_sample_num != y_sample_num:
            raise SampleNumberMismatchError
        self.sample_num = x_sample_num
        self.variable_num = x.shape[1]
        self.x = np.array(x)
        self.y = y
        self.label_type = self._check_label_type(self.y)
        self.feature_type = self._check_feature_type(self.x)

    def _clear(self):
        self.x = None
        self.y = None
        self.sample_num = 0
        self.variable_num = 0

    @staticmethod
    def _check_x(x):
        if isinstance(x, np.ndarray):
            if len(x.shape) != 2:
                raise FeatureTypeError("请输入二维数组")
            return x.shape[0]
        else:
            raise FeatureTypeError("请输入二维Numpy.array或pandas.Series")

    @staticmethod
    def _check_y(y):
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise LabelArrayTypeError("请输入一维数组")
            return y.shape[0]
        else:
            raise LabelArrayTypeError("请输入一维Numpy.array或pandas.Series")

    @staticmethod
    def _check_label_type(y):
        count = np.unique(y)
        if len(count) == 2:
            return LabelType.binary
        elif len(count) > len(y) / 2:
            return LabelType.continuous
        else:
            return LabelType.multi_class

    @staticmethod
    def _check_feature_type(x):
        res = []
        for feature in x.T:
            count = np.unique(feature)
            if len(count) == 2:
                res.append(LabelType.binary)
            elif len(count) > len(feature) // 2:
                res.append(LabelType.continuous)
            else:
                res.append(LabelType.multi_class)
        return res

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def fit_transform(self, x, y):
        pass

class BaseFeatureSelect():

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def feature_select(self, top_n):
        pass


class BinaryTreeNode:

    def __init__(self, left=None, right=None, data_id=None, feature_id=None, value=None, leaf_label=None):
        if data_id is None:
            data_id = []
        self.left = left
        self.right = right
        self.data_id = data_id
        self.feature_id = feature_id
        self.value = value
        self.leaf_label = leaf_label


class MultiTreeNode:

    def __init__(self, child=None, data_id=None, feature_id=None, value=None, leaf_label=None):
        if data_id is None:
            data_id = []
        self.child = child
        self.data_id = data_id
        self.feature_id = feature_id
        self.value = value
        self.leaf_label = leaf_label


class GBDTTreeNode(BinaryTreeNode):

    def __init__(self, left=None, right=None, data_id=None, feature_id=None, value=None, leaf_label=None):
        super(GBDTTreeNode, self).__init__(left, right, data_id, feature_id, value, leaf_label)
        # gamma: GBDT需要记录的元素，当是回归树（平方损失）时，等于y_predict
        # 只有叶节点需要记录
        # gama反映的是这个叶子节点最优的预测值
        self.gamma = None
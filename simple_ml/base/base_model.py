# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

from abc import ABCMeta, abstractmethod
import numpy as np
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.data_handle import get_type
from collections import defaultdict


__all__ = [
    'BaseModel',
    'BaseClassifier',
    'BaseFeatureSelect',
    'BaseTransform',
    'Multi2Binary',
    'BinaryTreeNode',
    'MultiTreeNode',
    'GBDTTreeNode'
]


class BaseModel(object):

    def __init__(self):
        self.y_dic = {}    # 注意： 不能放在__init__(self)上面，否则成为类变量（类属性），所有实例将会共享

    def _init(self, x, y):
        self._clear()
        if y is not None:
            x_sample_num = self._check_x(x)
            y_sample_num = self._check_y(y)
            if x_sample_num != y_sample_num:
                raise SampleNumberMismatchError
            self.sample_num = x_sample_num
            self.variable_num = x.shape[1]
            self.x = np.array(x)
            self.label_type = self._check_label_type(y)

            # 统一分类标签的名称
            _min = min(y)
            for i in np.unique(y):
                self.y_dic[i] = i - _min

            # 如果是连续型，则不变，如果是离散型，则进行统一
            if self.label_type == LabelType.continuous:
                self.y = y
            else:
                self.y = np.array([self.y_dic[i] for i in y])

            self.feature_type = self._check_feature_type(self.x)
        else:
            self.sample_num = self._check_x(x)
            self.variable_num = x.shape[1]
            self.x = np.array(x)
            self.y = None
            self.label_type = None
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

        else:
            raise FeatureTypeError("请输入二维Numpy.array或pandas.Series")

        if not np.isfinite(x).all():
            raise ArrayContainNANorINF("input:X ")

        return x.shape[0]

    @staticmethod
    def _check_y(y):
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise LabelArrayTypeError("请输入一维数组")
        else:
            raise LabelArrayTypeError("请输入一维Numpy.array或pandas.Series")

        if not np.isfinite(y).all():
            raise ArrayContainNANorINF("input:y ")

        return y.shape[0]

    @staticmethod
    def _check_label_type(y):
        return get_type(y)

    @staticmethod
    def _check_feature_type(x):
        return get_type(x)

    def _check_x_test(self, x):
        self._check_x(x)
        if x.shape[1] != self.variable_num:
            raise FeatureNumberMismatchError

    def _check_y_test(self, y):
        if y is not None:
            self._check_y(y)
            y_type = self._check_label_type(y)
            if y_type != self.label_type:
                raise LabelTypeError("测试集标签类型必须和训练集一致")

    def check_test_data(self, x, y=None):
        self._check_x_test(x)
        if y is not None:
            self._check_y_test(y)
            for i in range(len(y)):
                y[i] = self.y_dic[y[i]]
            if y.shape[0] != x.shape[0]:
                raise LabelLengthMismatchError


class BaseClassifier(BaseModel):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._function = Function.classify
        super(BaseClassifier, self).__init__()

    @abstractmethod
    def fit(self, x, y):
        self._init(x, y)

    @abstractmethod
    def predict(self, x):
        self.check_test_data(x)

    @abstractmethod
    def score(self, x, y):
        self.check_test_data(x, y)

    @abstractmethod
    def new(self):
        pass


class BaseTransform(BaseModel):

    __doc__ = "This is a Transform Abstract Class"

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseTransform, self).__init__()

    def _init(self, x, y=None):
        super(BaseTransform, self)._init(x, y)

    @abstractmethod
    def fit(self, x, y):
        self._init(x, y)

    @abstractmethod
    def transform(self, x):
        self.check_test_data(x)

    @abstractmethod
    def fit_transform(self, x, y):
        self.check_test_data(x, y)


class BaseFeatureSelect:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def feature_select(self, top_n):
        pass


class Multi2Binary:

    def __init__(self):
        self.new_models = None
        self.y_unique = None
        self.model_num = None

    def _multi_fit(self, model):
        self.y_unique = np.unique(model.y)
        self.model_num = len(self.y_unique)
        self.new_models = [model.new() for i in range(self.model_num)]
        for i, y_value in enumerate(self.y_unique):
            new_y = (model.y == y_value).astype('int')
            self.new_models[i].fit(model.x, new_y)

    def _multi_predict_single(self, x):
        dic = defaultdict(int)

        for i in range(self.model_num):
            y_predict = self.new_models[i].predict(x.reshape(1, -1))
            if y_predict == 1:
                dic[i] += 1
            else:
                # 如果第i个类别分类是0，则以为其他所有类别得分都加1
                for j in range(self.model_num):
                    if j != i:
                        dic[j] += 1
        return max(dic.keys(), key=lambda x: dic[x])

    def _multi_predict(self, x):
        return np.array([self._multi_predict_single(i) for i in x])


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

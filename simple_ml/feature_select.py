# -*- coding:utf-8 -*-

"""
filter方法进行特征选择
"""
from __future__ import division, absolute_import

import numpy as np
from simple_ml.base.base_enum import FilterType, LabelType, EmbeddedType
from simple_ml.base.base_error import *
from simple_ml.base.base_model import BaseTransform


__all__ = ['Filter', 'Embedded', 'FilterType', 'EmbeddedType']


class Filter(BaseTransform):

    def __init__(self, filter_type, top_k):
        super(Filter, self).__init__()
        self.filterType = filter_type
        self.top_k = top_k

    def _var_select(self):
        variance_array = list(map(np.var, self.x.T))
        self._get_top_k_ids(variance_array)

    def _corr_select(self):
        corr_array = list(map(lambda x: np.corrcoef(x, self.y)[0][1], self.x.T))
        self._get_top_k_ids(corr_array)

    def _chi_select(self):
        """
        卡方检验用以检验两个事件是否独立，ref: http://blog.csdn.net/yihucha166/article/details/50646615
        检验统计量越大表示越有相关性
        """
        from sklearn.feature_selection import chi2
        chi2_array = chi2(self.x, self.y)[0]
        self._get_top_k_ids(chi2_array)

    def _entropy_select(self):
        """
        互信息法
        from minepy import MINE
        """
        pass
    #     m = MINE()
    #     mic_array = np.zeros(self.sample_num)
    #     for i, x in enumerate(self.x.T):
    #         m.compute_score(x, self.y)
    #         mic_array[i] = m.mic()
    #     self._get_top_k_ids(mic_array)

    def _get_top_k_ids(self, value_array, choose_max=True):
        """
        根据输入的数组，得到前k个最大或者是最小值对应的ids
        如果是要选择对应值最大的，则默认参数，否则改为False
        """
        pair = list(zip(range(self.variable_num), value_array))

        if choose_max:
            pair = sorted(pair, key=lambda x: x[1], reverse=True)
        else:
            pair = sorted(pair, key=lambda x: x[1])

        self._selectIds = np.array(list(zip(*pair))[0][:self.top_k])

    def fit(self, x, y=None):
        """
        返回当前选择x的列编号
        - 输入x要么是连续，要么是0-1，不存在多分类
        - 输入y可以使连续，0-1或是多分类
        """
        self._init(x, y)
        if self.filterType == FilterType.var:
            self._var_select()
        elif self.filterType == FilterType.corr:
            if self.label_type == LabelType.multi_class:
                raise LabelTypeError
            self._corr_select()
        elif self.filterType == FilterType.chi2:
            if self.label_type == LabelType.continuous:
                raise LabelTypeError
            self._chi_select()
        elif self.filterType == FilterType.entropy:
            self._entropy_select()

    def transform(self, x):
        return x[:, self._selectIds]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


class Embedded(BaseTransform):

    def __init__(self, top_k, embedded_type=EmbeddedType.Lasso):
        super(Embedded, self).__init__()
        self.embedded_type = embedded_type
        self.top_k = top_k
        self.model = None

    def fit(self, x, y):
        self._init(x, y)
        if self.embedded_type == EmbeddedType.Lasso:
            from simple_ml.logistic import Lasso
            self.model = Lasso()
        elif self.embedded_type == EmbeddedType.GBDT:
            from simple_ml.ensemble import GBDT
            self.model = GBDT()
        else:
            raise EmbeddedTypeError

        self.model.fit(x, y)
        self.selected_feature_id = self.model.feature_select(self.top_k)

    def transform(self, x):
        return x[:, self.selected_feature_id]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

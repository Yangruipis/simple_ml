# -*- coding:utf-8 -*-

"""
filter方法进行特征选择
"""

from minepy import MINE
from .my_classifier import *
from collections import Counter
from .my_enumrate import FilterType
from .my_error import *


class MyFilter(object):

    def __init__(self, filter_type, top_k):
        self.filterType = filter_type
        self.top_k = top_k

    def _var_select(self):
        variance_array = list(map(lambda x: np.var(x), self.x.T))
        self._get_top_k_ids(variance_array)

    def _corr_select(self):
        func = lambda x, y: np.corrcoef(x, y)
        corr_array = list(map(lambda x: func(x, self.y)[0][1], self.x.T))
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
        """
        m = MINE()
        mic_array = np.zeros(self.sample_num)
        for i, x in enumerate(self.x.T):
            m.compute_score(x, self.y)
            mic_array[i] = m.mic()
        self._get_top_k_ids(mic_array)

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

    @staticmethod
    def _check_label_type(y):
        count = dict(Counter(y))
        if len(count) == 2:
            return LabelType.binary
        elif len(count) > len(y)/2:
            return LabelType.continuous
        else:
            return LabelType.multiclass

    def _check(self, x, y):
        if x.shape[0] == y.shape[0]
            raise SampleNumberMismatchError
        if len(y.shape) == 1:
            raise LabelTypeError
        self.label_type = self._check_label_type(y)
        self.x = x
        self.y = y
        self.sample_num = x.shape[0]
        self.variable_num = x.shape[1]
        if self.top_k < self.variable_num:
            raise ValueBoundaryError
        self._selectIds = np.zeros(self.top_k)

    def fit(self, x, y=None):
        """
        返回当前选择x的列编号
        - 输入x要么是连续，要么是0-1，不存在多分类
        - 输入y可以使连续，0-1或是多分类
        """
        self._check(x, y)
        if self.filterType == FilterType.var:
            self._var_select()
        elif self.filterType == FilterType.corr:
            if self.label_type == LabelType.multiclass:
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

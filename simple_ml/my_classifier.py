# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from collections import Counter
from .my_enumrate import *
from .my_error import *


class MyClassifier(object):

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

    def _clear(self):
        self.x = None
        self.y = None
        self.sample_num = 0
        self.variable_num = 0

    @staticmethod
    def _check_x(x):
        if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame):
            if len(x.shape) != 2:
                raise FeatureTypeError
            return x.shape[0]
        else:
            raise FeatureTypeError

    @staticmethod
    def _check_y(y):
        if isinstance(y, np.ndarray) or isinstance(y, pd.Series):
            if len(y.shape) != 1:
                raise LabelArrayTypeError
            return y.shape[0]
        else:
            raise LabelArrayTypeError

    @staticmethod
    def _check_label_type(y):
        count = dict(Counter(y))
        if len(count) == 2:
            return LabelType.binary
        elif len(count) > len(y)/2:
            return LabelType.continuous
        else:
            return LabelType.multiclass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def score(self, x, y):
        pass

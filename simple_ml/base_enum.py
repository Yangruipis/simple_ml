# -*- coding:utf-8 -*-

from enum import Enum


class DisType(Enum):
    Eculidean = 1
    Manhattan = 2
    Minkowski = 3               # 明可夫斯基距离
    Chebyshev = 4               # 切比雪夫距离
    CosSim = 5                  # 余弦角距离


class CrossValidationType(Enum):
    holdout = 0
    k_folder = 1


class FilterType(Enum):
    var = 0
    corr = 1
    chi2 = 3
    entropy = 4


class LabelType(Enum):
    binary = 1
    multi_class = 2
    continuous = 3


class KernelType(Enum):
    linear = 0
    polynomial = 1
    gassian = 2
    laplace = 3
    sigmoid = 4

class ClassifierType(Enum):
    LR = 0
    CART = 1
    SVM = 2
    NB = 3
    KNN = 4
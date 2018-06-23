# -*- coding:utf-8 -*-

from enum import Enum

__all__ = [
    'DisType',
    'CrossValidationType',
    'FilterType',
    'LabelType',
    'KernelType',
    'ClassifierType',
    'EmbeddedType',
    'ConMissingHandle',
    'DisMissingHandle',
    'CostFunction',
    'ActiveFunction',
    'OptMethod',
    'Function',
]


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
    gaussian = 2
    laplace = 3
    sigmoid = 4


class ClassifierType(Enum):
    LR = 0
    CART = 1
    SVM = 2
    NB = 3
    KNN = 4


class EmbeddedType(Enum):
    GBDT = 0
    Lasso = 1


class ConMissingHandle(Enum):
    """
    连续数据缺失值处理方法
    """
    mean_fill = 0
    median_fill = 1
    sample_drop = 2


class DisMissingHandle(Enum):
    """
    离散数据缺失值处理方法
    """
    mode_fill = 0
    sample_drop = 1
    one_hot = 2


class CostFunction(Enum):

    square = 0
    logistic = 1
    hinge = 2        # max(0, 1 - t*y)，t为目标值，y为输出值
    softmax = 3


class ActiveFunction(Enum):

    sigmoid = 0
    tanh = 1
    relu = 2


class OptMethod(Enum):

    SA = 0
    PSO = 1
    down_hill = 2
    grid_search = 3
    bayes = 4


class Function(Enum):

    classify = 1
    regression = 2
    cls_and_reg = 3
    cluster = 4

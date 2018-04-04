# -*- coding:utf-8 -*-


class MisMatchError(Exception):

    def __init__(self, err="不匹配"):
        super(MisMatchError, self).__init__(err)


class FeatureNumberMismatchError(MisMatchError):

    def __init__(self, err="特征数目不匹配"):
        super(FeatureNumberMismatchError, self).__init__(err)


class LabelLengthMismatchError(MisMatchError):

    def __init__(self, err="标签长度不匹配"):
        super(LabelLengthMismatchError, self).__init__(err)


class TreeNumberMismatchError(MisMatchError):

    def __init__(self, err="树数目不匹配"):
        super(TreeNumberMismatchError, self).__init__(err)


class SampleNumberMismatchError(MisMatchError):

    def __init__(self, err="样本数目不匹配"):
        super(SampleNumberMismatchError, self).__init__(err)


class CostMatMismatchError(MisMatchError):

    def __init__(self, err="损失矩阵维度不匹配"):
        super(CostMatMismatchError, self).__init__(err)


class FeatureTypeError(TypeError):
    pass


class LabelArrayTypeError(TypeError):
    pass


class DistanceTypeError(TypeError):
    pass


class CrossValidationTypeError(TypeError):
    pass


class FilterTypeError(TypeError):
    pass


class LabelTypeError(TypeError):
    pass


class KernelTypeError(TypeError):
    pass


class ValueBoundaryError(ValueError):
    pass


class KernelMissParameterError(KernelTypeError):
    pass


class ModelNotFittedError(Exception):
    pass

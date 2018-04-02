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

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class LabelArrayTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class DistanceTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class CrossValidationTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class FilterTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class LabelTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class KernelTypeError(TypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class ValueBoundaryError(ValueError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class KernelMissParameterError(KernelTypeError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class ModelNotFittedError(Exception):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass

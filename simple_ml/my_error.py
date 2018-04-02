# -*- coding:utf-8 -*-


class MisMatchError(Exception):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class FeatureNumberMismatchError(MisMatchError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class LabelLengthMismatchError(MisMatchError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class TreeNumberMismatchError(MisMatchError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class SampleNumberMismatchError(MisMatchError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class CostMatMismatchError(MisMatchError):

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


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

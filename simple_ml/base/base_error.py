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


class TopNTooLargeError(MisMatchError):

    def __init__(self, err="Top_N 数目超过可选择的最大数目"):
        super(TopNTooLargeError, self).__init__(err)


class FeatureTypeError(TypeError):
    pass


class ClassifierTypeError(TypeError):
    pass


class LabelArrayTypeError(TypeError):
    pass


class DistanceTypeError(TypeError):
    pass


class CrossValidationTypeError(TypeError):
    pass


class FilterTypeError(TypeError):
    pass

class EmbeddedTypeError(TypeError):
    pass

class LabelTypeError(TypeError):
    pass


class KernelTypeError(TypeError):
    pass


class ValueBoundaryError(ValueError):
    pass


class NeedMoreSampleError(ValueError):
    pass


class KernelMissParameterError(KernelTypeError):
    pass


class ModelNotFittedError(Exception):
    pass


class EmptyInputError(ValueError):
    pass


class MissingHandleTypeError(TypeError):
    pass

class CostFunctionError(TypeError):
    pass

class NeuralNetworkParamError(ValueError):
    pass
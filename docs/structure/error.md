# 异常类 simple_ml.base.base_error

- [异常类 simple_ml.base.base_error](#%E5%BC%82%E5%B8%B8%E7%B1%BB-simplemlbasebaseerror)
    - [异常：不匹配 (MisMatchError)](#%E5%BC%82%E5%B8%B8%EF%BC%9A%E4%B8%8D%E5%8C%B9%E9%85%8D-mismatcherror)
        - [特征数目不匹配 (FeatureNumberMismatchError)](#%E7%89%B9%E5%BE%81%E6%95%B0%E7%9B%AE%E4%B8%8D%E5%8C%B9%E9%85%8D-featurenumbermismatcherror)
        - [标签长度不匹配 (LabelLengthMismatchError)](#%E6%A0%87%E7%AD%BE%E9%95%BF%E5%BA%A6%E4%B8%8D%E5%8C%B9%E9%85%8D-labellengthmismatcherror)
        - [树数目不匹配 (TreeNumberMismatchError)](#%E6%A0%91%E6%95%B0%E7%9B%AE%E4%B8%8D%E5%8C%B9%E9%85%8D-treenumbermismatcherror)
        - [样本数目不匹配 (SampleNumberMismatchError)](#%E6%A0%B7%E6%9C%AC%E6%95%B0%E7%9B%AE%E4%B8%8D%E5%8C%B9%E9%85%8D-samplenumbermismatcherror)
        - [损失矩阵维度不匹配 (CostMatMismatchError)](#%E6%8D%9F%E5%A4%B1%E7%9F%A9%E9%98%B5%E7%BB%B4%E5%BA%A6%E4%B8%8D%E5%8C%B9%E9%85%8D-costmatmismatcherror)
        - [前N大数目越界 (TopNTooLargeError)](#%E5%89%8Dn%E5%A4%A7%E6%95%B0%E7%9B%AE%E8%B6%8A%E7%95%8C-topntoolargeerror)
    - [异常：类型错误 (TypeError)](#%E5%BC%82%E5%B8%B8%EF%BC%9A%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-typeerror)
        - [特征类型错误 (FeatureTypeError)](#%E7%89%B9%E5%BE%81%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-featuretypeerror)
        - [分类器类型错误 (ClassifierTypeError)](#%E5%88%86%E7%B1%BB%E5%99%A8%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-classifiertypeerror)
        - [距离类型错误 (DistanceTypeError)](#%E8%B7%9D%E7%A6%BB%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-distancetypeerror)
        - [交叉验证类型错误 (CrossValidationTypeError)](#%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-crossvalidationtypeerror)
        - [特征选择方法类型错误 (FilterTypeError)](#%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%E6%96%B9%E6%B3%95%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-filtertypeerror)
        - [特征选择方法类型错误 (EmbeddedTypeError)](#%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%E6%96%B9%E6%B3%95%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-embeddedtypeerror)
        - [标签类型错误 (LabelTypeError)](#%E6%A0%87%E7%AD%BE%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-labeltypeerror)
        - [核函数类型错误 (KernelTypeError)](#%E6%A0%B8%E5%87%BD%E6%95%B0%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-kerneltypeerror)
        - [核函数缺失参数 (KernelMissParameterError)](#%E6%A0%B8%E5%87%BD%E6%95%B0%E7%BC%BA%E5%A4%B1%E5%8F%82%E6%95%B0-kernelmissparametererror)
        - [标签数组类型错误 (LabelArrayTypeError)](#%E6%A0%87%E7%AD%BE%E6%95%B0%E7%BB%84%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-labelarraytypeerror)
        - [缺失值处理类型错误 (MissingHandleTypeError)](#%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86%E7%B1%BB%E5%9E%8B%E9%94%99%E8%AF%AF-missinghandletypeerror)
    - [异常：值错误 (ValueError)](#%E5%BC%82%E5%B8%B8%EF%BC%9A%E5%80%BC%E9%94%99%E8%AF%AF-valueerror)
        - [值越界 (ValueBoundaryError)](#%E5%80%BC%E8%B6%8A%E7%95%8C-valueboundaryerror)
        - [样本不够 (NeedMoreSampleError)](#%E6%A0%B7%E6%9C%AC%E4%B8%8D%E5%A4%9F-needmoresampleerror)
        - [空输入 (EmptyInputError)](#%E7%A9%BA%E8%BE%93%E5%85%A5-emptyinputerror)
    - [其他异常](#%E5%85%B6%E4%BB%96%E5%BC%82%E5%B8%B8)
    - [ModelNotFittedError 模型未训练](#modelnotfittederror-%E6%A8%A1%E5%9E%8B%E6%9C%AA%E8%AE%AD%E7%BB%83)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *

## 异常：不匹配 (MisMatchError)

```python
class MisMatchError(Exception):

    def __init__(self, err="不匹配"):
        super(MisMatchError, self).__init__(err)
```
### 特征数目不匹配 (FeatureNumberMismatchError)

```python
class FeatureNumberMismatchError(MisMatchError):

    def __init__(self, err="特征数目不匹配"):
        super(FeatureNumberMismatchError, self).__init__(err)
```

### 标签长度不匹配 (LabelLengthMismatchError)

```python
class LabelLengthMismatchError(MisMatchError):

    def __init__(self, err="标签长度不匹配"):
        super(LabelLengthMismatchError, self).__init__(err)
```

### 树数目不匹配 (TreeNumberMismatchError)

```python
class TreeNumberMismatchError(MisMatchError):

    def __init__(self, err="树数目不匹配"):
        super(TreeNumberMismatchError, self).__init__(err)
```

### 样本数目不匹配 (SampleNumberMismatchError)

```python
class SampleNumberMismatchError(MisMatchError):

    def __init__(self, err="样本数目不匹配"):
        super(SampleNumberMismatchError, self).__init__(err)
```

### 损失矩阵维度不匹配 (CostMatMismatchError)

```python
class CostMatMismatchError(MisMatchError):

    def __init__(self, err="损失矩阵维度不匹配"):
        super(CostMatMismatchError, self).__init__(err)
```

### 前N大数目越界 (TopNTooLargeError)

```python
class TopNTooLargeError(MisMatchError):

    def __init__(self, err="top n数目超过可选择的最大数目"):
        super(TopNTooLargeError, self).__init__(err)
```


## 异常：类型错误 (TypeError)

### 特征类型错误 (FeatureTypeError)

### 分类器类型错误 (ClassifierTypeError)

### 距离类型错误 (DistanceTypeError)

### 交叉验证类型错误 (CrossValidationTypeError)

### 特征选择方法类型错误 (FilterTypeError)

### 特征选择方法类型错误 (EmbeddedTypeError)

### 标签类型错误 (LabelTypeError)

### 核函数类型错误 (KernelTypeError)

### 核函数缺失参数 (KernelMissParameterError)

### 标签数组类型错误 (LabelArrayTypeError)

### 缺失值处理类型错误 (MissingHandleTypeError)


## 异常：值错误 (ValueError)

### 值越界 (ValueBoundaryError)
### 样本不够 (NeedMoreSampleError)
### 空输入 (EmptyInputError)


## 其他异常

## ModelNotFittedError 模型未训练

# [返回主页](../index.md)

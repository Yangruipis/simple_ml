# 枚举类 simple_ml.base.base_enum

- [枚举类 simple_ml.base.base_enum](#%E6%9E%9A%E4%B8%BE%E7%B1%BB-simplemlbasebaseenum)
    - [距离类型 DisType](#%E8%B7%9D%E7%A6%BB%E7%B1%BB%E5%9E%8B-distype)
    - [交叉验证类型 CrossValidationType](#%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E7%B1%BB%E5%9E%8B-crossvalidationtype)
    - [Filter特征选择类型 FilterType](#filter%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%E7%B1%BB%E5%9E%8B-filtertype)
    - [变量类型 LabelType](#%E5%8F%98%E9%87%8F%E7%B1%BB%E5%9E%8B-labeltype)
    - [核函数类型 KernelType](#%E6%A0%B8%E5%87%BD%E6%95%B0%E7%B1%BB%E5%9E%8B-kerneltype)
    - [分类器类别 ClassifierType](#%E5%88%86%E7%B1%BB%E5%99%A8%E7%B1%BB%E5%88%AB-classifiertype)
    - [Embedded特征选择类型 EmbeddedType](#embedded%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%E7%B1%BB%E5%9E%8B-embeddedtype)
    - [连续数据缺失值处理类型 ConMissingHandle](#%E8%BF%9E%E7%BB%AD%E6%95%B0%E6%8D%AE%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86%E7%B1%BB%E5%9E%8B-conmissinghandle)
    - [离散数据缺失值处理类型 DisMissingHandle](#%E7%A6%BB%E6%95%A3%E6%95%B0%E6%8D%AE%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86%E7%B1%BB%E5%9E%8B-dismissinghandle)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *

## 距离类型 DisType

```python
class DisType(Enum):
    Eculidean = 1               # 欧几里得距离
    Manhattan = 2               # 曼哈顿距离
    Minkowski = 3               # 明可夫斯基距离
    Chebyshev = 4               # 切比雪夫距离
    CosSim = 5                  # 余弦角距离
```


## 交叉验证类型 CrossValidationType

```python
class CrossValidationType(Enum):
    holdout = 0                 # 留出法交叉验证
    k_folder = 1                # k折交叉验证
```

## Filter特征选择类型 FilterType

```python
class FilterType(Enum):
    var = 0                     # 方差法
    corr = 1                    # 相关系数法 
    chi2 = 3                    # 卡方检验法
    entropy = 4                 # 互信息法
```

## 变量类型 LabelType

```python
class LabelType(Enum):
    binary = 1                  # 二值变量
    multi_class = 2             # 多值变量
    continuous = 3              # 连续变量
```

## 核函数类型 KernelType

```python
class KernelType(Enum):
    linear = 0                  # 线性核
    polynomial = 1              # 多项式核
    gaussian = 2                # 高斯核
    laplace = 3                 # 拉普拉斯核
    sigmoid = 4                 # sigmoid核
```

## 分类器类别 ClassifierType

```python
class ClassifierType(Enum):
    LR = 0                      # 逻辑回归分类器
    CART = 1                    # 分类回归树（CART）分类器
    SVM = 2                     # 支持向量机分类器
    NB = 3                      # 朴素贝叶斯分类器
    KNN = 4                     # K近邻分类器
```

## Embedded特征选择类型 EmbeddedType

```python
class EmbeddedType(Enum):
    GBDT = 0                    # GBDT特征选择
    Lasso = 1                   # Lasso特征选择
```

## 连续数据缺失值处理类型 ConMissingHandle

```python
class ConMissingHandle(Enum):
    mean_fill = 0               # 均值填补
    median_fill = 1             # 中位数填补
    sample_drop = 2             # 抛弃缺失样本
```

## 离散数据缺失值处理类型 DisMissingHandle

```python
class DisMissingHandle(Enum):
    mode_fill = 0               # 中位数填补
    sample_drop = 1             # 抛弃缺失样本
    one_hot = 2                 # 独热编码，将缺失值作为一个新的类比，进行独热编码
```



# [返回主页](../index.md)


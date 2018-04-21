# PCA降维模块 **simple_ml.pca**


## 主成分分析 (PCA)

```python
from simple_ml.base.base_model import BaseTransform

class PCA(BaseTransform):

    def __init__(self, top_n):
        pass
```

`simple_ml`提供了常规的主成分分析方法进行降维

* * *

**初始化**

|             | 名称  | 类型 |      描述       |
|------------:|:-----:|:----:|:--------------:|
| Parameters: | top_n | int  | 希望保留几个维度 |


**类方法**
1. 

```python
def fit(self, x, y=None)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     特征      |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      |    Void    |              |


1. 

```python
def transform(self, x)
```

将x转换为PCA降维之后的形式，输出维度为(n,
top_n)，`n`为样本数，`top_n`为初始化值

|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|    Returns: |      | np.2darray | 特征选择后的数组 |

1. 

```python
def fit_transform(self, x, y)
```

拟合并且转换为处理后的数组，等于`fit`+`transform`


|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|             |  y   |  np.array  |  标签，可以没有  |
|    Returns: |      | np.2darray | 特征选择后的数组 |


**类属性**

|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| explain_ratio |   float    | 前top_n个主成分的解释比 |
|  eigen_value  |  np.array  |    训练数据的特征值    |
| eigen_vecttor | np.2darray |   训练数据的特征向量    |


## 针对高维数据的主成分分析 (SuperPCA)

```python
from simple_ml.pca import PCA

class SuperPCA(PCA):

    def __init__(self, top_n):
        """
        针对数据维度大于样本数目的情况，可以通过矩阵分解简化计算，
        但是最多只能得到等同于样本数目的主成分个数：
            Pv = lambda v
            XX'v = lambda v
            X'XX'v = X'lambda v
            sigma x' v = lambda X'v
            sigma (X'v) = lambda (X'v)
        所以只要求 XX'的主成分即可，其中X为去每列减去均值后除以根号（总行数-1），
        保证协方差矩阵sigma等于 X'X
        :param top_n: 主成分个数，当大于样本个数时报错
        """
        super(SuperPCA, self).__init__(top_n)
```

当数据维度较高，且样本较少，尤其是遇到图像降维的问题时，
`simple_ml`提供了针对高维数据的降维方法

优点：
- 速度非常快
- 无需太多样本，即可得到很好的结果

缺点：
- 只能得到不大于样本数的主成分个数


* * *

**初始化**

|             | 名称  | 类型 |      描述       |
|------------:|:-----:|:----:|:--------------:|
| Parameters: | top_n | int  | 希望保留几个维度 |


**类方法**
1. 

```python
def fit(self, x, y=None)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     特征      |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      |    Void    |              |


1. 

```python
def transform(self, x)
```

将x转换为SuperPCA降维之后的形式，输出维度为(n,
top_n)，`n`为样本数，`top_n`为初始化值

|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|    Returns: |      | np.2darray | 特征选择后的数组 |

1. 

```python
def fit_transform(self, x, y)
```

拟合并且转换为处理后的数组，等于`fit`+`transform`


|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|             |  y   |  np.array  |  标签，可以没有  |
|    Returns: |      | np.2darray | 特征选择后的数组 |


**类属性**

|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| explain_ratio |   float    | 前top_n个主成分的解释比 |
|  eigen_value  |  np.array  |    训练数据的特征值    |
| eigen_vecttor | np.2darray |   训练数据的特征向量    |


# [返回主页](../index.md)



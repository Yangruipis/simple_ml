
# 特征选择模块 **simple_ml.feature_select**


- [特征选择模块 **simple_ml.feature_select**](#%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%E6%A8%A1%E5%9D%97-simplemlfeatureselect)
    - [Filter方法](#filter%E6%96%B9%E6%B3%95)
    - [Embedded方法](#embedded%E6%96%B9%E6%B3%95)
- [返回主页](#%E6%94%BE%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *

## Filter方法

- `simple_ml`提供了Filter特征选择方法
- Filter是指通过特征与标签之间的关系来进行特征选择
- 目前支持的Filter方法有：
  - 方差法，即认为方差越大的特征反映的信息越多
  - 相关系数法（只针对连续变量），特征与标签相关性越强，越要保留
  - 卡方检验法（只针对离散变量），同样通过特征与标签的关系进行特征选择
  - 互信息法（均可），同上


```python
from simple_ml.base.base_model import BaseTransform
from simple_ml.base.base_enum import FilterType


class Filter(BaseTransform):

    def __init__(self, top_k, filter_type=FilterType.var):
        pass
```

**初始化**

|             |    名称     |                类型                |     描述      |
|------------:|:-----------:|:----------------------------------:|:-------------:|
| Parameters: | filter_type | [FilterType](../structure/enum.md) |  Filter类型   |
|             |    tok_k    |                int                 | 希望取几个特征 |


* * *

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


2.

```python
def transform(self, x)
```

转换特征，转为特征选择之后的形式，输出维度为(n, top_k)，`n`为样本数，`top_k`为初始化值

|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|    Returns: |      | np.2darray | 特征选择后的数组 |

3.

```python
def fit_transform(self, x, y)
```

拟合并且转换为处理后的数组


|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      | np.2darray | 特征选择后的数组 |


* * *

**类属性**

None

* * *

## Embedded方法

Embedded特征选择方法通过模型自身的特性进行选择，`simple_ml`提供了两种特征选择方法
- Lasso，通过L1正则项约束下参数系数化的特性进行选择
- GBDT， 通过每一曾树可以对特征重要性进行打分（节点对损失函数的降低程度），从而进行特征选择

`注：` Lasso用于离散特征，GBDT用于连续特征


```python
from simple_ml.base.base_model import BaseTransform
from simple_ml.base.base_enum import EmbeddedType

class Embedded(BaseTransform):

    def __init__(self, top_k, embedded_type=EmbeddedType.Lasso):
        pass    
```



**初始化**

|             |    名称     |                类型                |     描述      |
|------------:|:-----------:|:----------------------------------:|:-------------:|
| Parameters: | embedded_type | [EmbeddedType](../structure/enum.md) |  Embedded类型   |
|             |    tok_k    |                int                 | 希望取几个特征 |


* * *

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


2.

```python
def transform(self, x)
```

转换特征，转为特征选择之后的形式，输出维度为(n, top_k)，`n`为样本数，`top_k`为初始化值

|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|    Returns: |      | np.2darray | 特征选择后的数组 |

3.

```python
def fit_transform(self, x, y)
```

拟合并且转换为处理后的数组


|             | 名称 |    类型     |      描述       |
|------------:|:----:|:----------:|:--------------:|
| Parameters: |  x   | np.2darray |      特征       |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      | np.2darray | 特征选择后的数组 |


* * *

**类属性**

None

## Example

```python
from simple_ml.feature_select import Filter, Embedded
from simple_ml.base.base_enum import FilterType, EmbeddedType
import numpy as np
from simple_ml.classify_data import get_wine


x, y = get_wine()
x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

_filter = Filter(FilterType.corr, 3)
x_filter = _filter.fit_transform(x, y)
print(x_filter.shape)

_filter = Filter(FilterType.var, 3)
x_filter = _filter.fit_transform(x, y)
print(x_filter.shape)

_filter = Filter(FilterType.entropy, 3)
x_filter = _filter.fit_transform(x, y)
print(x_filter.shape)

embedded = Embedded(3, EmbeddedType.Lasso)
x_embedded = embedded.fit_transform(x, y)
print(x_embedded.shape) # lasso后稀疏到只有两个值非0，因此只输出了两个特征

# GBDT暂时只支持离散特征
embedded = Embedded(3, EmbeddedType.GBDT)
x = np.random.choice([0, 1], 50).reshape(10, 5)
y = np.random.rand(10)
x_embedded = embedded.fit_transform(x, y)
print(x_embedded, y)
```

# [返回主页](../index.md)
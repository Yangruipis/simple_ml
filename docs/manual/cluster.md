# 聚类模型 **simple_ml.cluster**


## 一、K均值聚类 (K-means Cluster)

```python
from simple_ml.base.base_enum import DisType


class KMeans(object):

    def __init__(self, k, dis_type=DisType.Eculidean, d=1):
         pass
```

`KMeans`模型是常规的聚类方法之一，通过EM算法迭代得到最优解

初始解随机获取（此处可优化为Kmeans++）

支持距离度量类型：
- 欧式距离
- 曼哈顿距离
- 明可夫斯基距离
- 契比雪夫距离
- 余弦角距离

### 1.1 初始化

|             |   名称   |              类型               |       描述       |
|------------:|:--------:|:-------------------------------:|:----------------:|
| Parameters: |    k     |               int               |      聚为k类      |
|             | dis_type | [DisType](../structure/enum.md) |   距离度量类型    |
|             |    d     |               int               | 明可夫斯基距离参数 |

### 1.2 类方法

1 拟合

```python
def fit(self, x)
```

拟合特征

|             | 名称 |    类型     | 描述 |
|------------:|:----:|:----------:|:---:|
| Parameters: |  x   | np.2darray | 特征 |
|    Returns: |      |    Void    |     |


### 1.3 类属性

|  名称  |   类型   |    描述     |
|:------:|:--------:|:-----------:|
| labels | np.array | 聚类后的标签 |


## 二、层次聚类 (Hierarchical Cluster)

```python
from simple_ml.base.base_enum import DisType

class Hierarchical(object):

    def __init__(self, dis_type=DisType.Eculidean, d=1):
        pass
```

层次聚类无需提供参数`k`，会聚成树状结构，需要给定最短距离之后输出


* * *

### 2.1 初始化

|            |   名称   |              类型               |       描述       |
|-----------:|:--------:|:-------------------------------:|:----------------:|
| Parameters | dis_type | [DisType](../structure/enum.md) |   距离度量类型    |
|            |    d     |               int               | 明可夫斯基距离参数 |


### 2.2 类方法


1 拟合

```python
def fit(self, x)
```

拟合特征

|             | 名称 |    类型     | 描述 |
|------------:|:----:|:----------:|:---:|
| Parameters: |  x   | np.2darray | 特征 |
|    Returns: |      |    Void    |     |


2 聚类

```python
def cluster(self, min_sim=None)
```

|             |  名称   |   类型   |                        描述                         |
|------------:|:-------:|:--------:|:---------------------------------------------------:|
| Parameters: | min_sim |  float   | 最小的相似度，从而截取聚类树，如果为None则取样本最大距离 |
|    Returns: |         | np.array |                  聚类结果，类别数组                   |


### 2.3 类属性


|  名称   |   类型   |      描述       |
|:-------:|:--------:|:---------------:|
| labels  | np.array |   聚类后的标签   |
| max_dis |  float   | 样本簇间最大距离 |


# [返回主页](../index.md)



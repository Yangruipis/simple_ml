# 聚类模型 **simple_ml.cluster**


- [聚类模型 **simple_ml.cluster**](#%E8%81%9A%E7%B1%BB%E6%A8%A1%E5%9E%8B-simplemlcluster)
    - [一、K均值聚类 (K-means Cluster)](#%E4%B8%80%E3%80%81k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB-k-means-cluster)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [二、层次聚类 (Hierarchical Cluster)](#%E4%BA%8C%E3%80%81%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB-hierarchical-cluster)
        - [2.1 初始化](#21-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [2.2 类方法](#22-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [2.3 类属性](#23-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [Examples](#examples)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)


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


## Examples

```python
from simple_ml.classify_data import get_iris
from simple_ml.cluster import *
import matplotlib.pyplot as plt


x, y = get_iris()

k_means = KMeans(k=4, dis_type=DisType.CosSim)
k_means.fit(x[:, :2])
print(k_means.labels)

plt.scatter(x=x[:, 0], y=x[:, 1], c=k_means.labels)
plt.show()

h_cluster = Hierarchical(dis_type=DisType.Manhattan)
h_cluster.fit(x[:, :2])
# 选取距离为最大距离的四分之一
h_cluster.cluster(h_cluster.max_dis/4)
print(h_cluster.labels)
plt.scatter(x=x[:, 0], y=x[:, 1], c=h_cluster.labels)
plt.show()
```

# [返回主页](../index.md)



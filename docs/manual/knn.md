# 邻近学习模块 **simple_ml.knn**

- [邻近学习模块 **simple_ml.knn**](#%E9%82%BB%E8%BF%91%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9D%97-simplemlknn)
    - [一、最近邻模型](#%E4%B8%80%E3%80%81%E6%9C%80%E8%BF%91%E9%82%BB%E6%A8%A1%E5%9E%8B)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [二、KD树（KNN的优化算法）](#%E4%BA%8C%E3%80%81kd%E6%A0%91%EF%BC%88knn%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%EF%BC%89)
    - [Example](#example)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 一、最近邻模型


```python
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import DisType

class KNN(BaseClassifier):

    __doc__ = "K Nearest Neighbor(s)"

    def __init__(self, k=1, distance_type=DisType.Eculidean):
        pass
```

`simple_ml` 提供了KNN类，支持距离类型包括：

- 欧几里得距离
- 曼哈顿距离
- 余弦角距离
- 切比雪夫距离

KNN类支持：
- 二分类问题
- 多分类问题

* * *

### 1.1 初始化

|             |     名称      |              类型               |          描述           |
|------------:|:-------------:|:-------------------------------:|:-----------------------:|
| Parameters: |       k       |               int               | 每次分类根据最近的k个样本 |
|             | distance_type | [DisType](../structure/enum.md) |         距离类型         |


### 1.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 训练集特征 |
|             |  y   |  np.array  | 训练集标签 |
|    Returns: |      |    Void    |           |


2 预测

```python
def predict(self, x)
```

给定测试集特征x，进行预测

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|    Returns: |      |  np.array  | 预测的结果 |

3 结果评价

```python
def score(self, x, y)
```

拟合并进行预测，最后给出预测效果的得分


|             | 名称 |    类型     |                            描述                            |
|------------:|:----:|:----------:|:---------------------------------------------------------:|
| Parameters: |  x   | np.2darray |                         测试集特征                         |
|             |  y   |  np.array  |                         测试集标签                         |
|    Returns: |      |   float    | 预测结果评分，如果是二分类则为F1值，如果是多分类则为Macro F1值 |

4 分类作图

`simple_ml` 提供了直接绘制分类效果图的方法，如果维度大于2，则通过PCA降至两维

```python
def classify_plot(self, x, y, title="")
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |
|    Returns: |      |    Void    |           |


### 1.3 类属性

None

## 二、KD树（KNN的优化算法）

Coming Soon


## Example

```python
from simple_ml.classify_data import get_wine
from simple_ml.data_handle import train_test_split
from simple_ml.knn import KNN

x, y = get_wine()
# knn可以解决多分类问题
# x = x[(y == 0) | (y == 1)]
# y = y[(y == 0) | (y == 1)]
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

knn = KNN()
knn.fit(x_train, y_train)
print(knn.score(x_test, y_test))
knn.classify_plot(x_test, y_test)

```

# [返回主页](../index.md)



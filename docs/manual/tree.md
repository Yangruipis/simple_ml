# 树模型 **simple_ml.tree**

- [树模型 **simple_ml.tree**](#%E6%A0%91%E6%A8%A1%E5%9E%8B-simplemltree)
    - [一、ID3算法](#%E4%B8%80%E3%80%81id3%E7%AE%97%E6%B3%95)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [二、分类回归树算法 (CART)](#%E4%BA%8C%E3%80%81%E5%88%86%E7%B1%BB%E5%9B%9E%E5%BD%92%E6%A0%91%E7%AE%97%E6%B3%95-cart)
        - [2.1 初始化](#21-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [2.2 类方法](#22-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [2.3 类属性](#23-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [Examples](#examples)
        - [二分类](#%E4%BA%8C%E5%88%86%E7%B1%BB)
        - [多分类](#%E5%A4%9A%E5%88%86%E7%B1%BB)
        - [回归](#%E5%9B%9E%E5%BD%92)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 一、ID3算法


```python
from simple_ml.base.base_model import BaseClassifier


class ID3(BaseClassifier):

    __doc__ = "ID3 Decision Tree"

    def __init__(self, max_depth=None, min_samples_leaf=3):
        """
        决策树ID3算法
        :param max_depth:        树最大深度
        :param min_samples_leaf: 叶子节点最大样本数（最好是奇数，用以投票）
        """
        pass
```

ID3模型，支持解决：
- 二分类问题
- 多分类问题

* * *

### 1.1 初始化

|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | max_depth | int  | 树最大深度         |
|     |    min_samples_leaf     | int | 叶子节点最大样本数（奇数）        |

### 1.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     特征      |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      |    Void    |              |


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
|    Returns: |      |   float    | 预测结果评分，二分类给出F1值，多分类给出Macro F1值 |

4 分类作图

绘制分类效果图，如果维度大于2，则通过PCA降至两维

```python
def classify_plot(self, x, y, title="")
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |
|    Returns: |      |    Void    |           |



### 1.3 类属性


|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| root |   [MultiTreeNode](../structure/struct.md)   |  树的根节点 |


## 二、分类回归树算法 (CART)

```python
from simple_ml.base.base_model import BaseClassifier


class CART(BaseClassifier):

    __doc__ = "Classify and Regression Tree"

    def __init__(self, max_depth=10, min_samples_leaf=5):
        """
        分类回归树
        :param max_depth:        树最大深度
        :param min_samples_leaf: 叶子节点最大样本数（最好是奇数，用以投票）
        """
        pass
```

CART模型，支持解决：
- 二分类问题
- 多分类问题
- 回归

* * *

### 2.1 初始化

|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | max_depth | int  | 树最大深度         |
|     |    min_samples_leaf     | int | 叶子节点最大样本数（奇数）        |

### 2.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     特征      |
|             |  y   |  np.array  | 标签，可以没有 |
|    Returns: |      |    Void    |              |


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
|    Returns: |      |   float    | 预测结果评分，二分类给出F1值，多分类给出Macro F1值 |

4 分类作图

绘制分类效果图，如果维度大于2，则通过PCA降至两维

```python
def classify_plot(self, x, y, title="")
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |
|    Returns: |      |    Void    |           |



### 2.3 类属性


|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| root |   [BinaryTreeNode](../structure/struct.md)   |  树的根节点 |

## Examples

### 二分类

```python
from simple_ml.classify_data import get_watermelon
from simple_ml.data_handle import train_test_split
from simple_ml.tree import ID3

x, y = get_watermelon()
x = x[:, :4]
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)
id3 = ID3()
id3.fit(x_train, y_train)
print(id3.score(x_test, y_test))
```

### 多分类

```python
from simple_ml.classify_data import get_wine
from simple_ml.data_handle import train_test_split
from simple_ml.tree import CART


x, y = get_wine()
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

cart = CART()
cart.fit(x_train, y_train)
print(cart.score(x_test, y_test))
cart.classify_plot(x_test, y_test)

```

### 回归

```python
from simple_ml.classify_data import get_wine
from simple_ml.data_handle import train_test_split
from simple_ml.tree import CART


x, y = get_wine()

y = x[:, -1]
x = x[:, :-1]
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

cart = CART()
cart.fit(x_train, y_train)
print(cart.score(x_test, y_test))
```

# [返回主页](../index.md)
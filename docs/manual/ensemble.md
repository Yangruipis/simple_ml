# 集成学习 **simple_ml.ensemble**

- [集成学习 **simple_ml.ensemble**](#%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0-simplemlensemble)
    - [一、AdaBoost算法](#%E4%B8%80%E3%80%81adaboost%E7%AE%97%E6%B3%95)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [二、梯度提升树 (GBDT)](#%E4%BA%8C%E3%80%81%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A0%91-gbdt)
        - [2.1 初始化](#21-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [2.2 类方法](#22-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [2.3 类属性](#23-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [随机森林 (RandomForest)](#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97-randomforest)
        - [3.1 初始化](#31-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [3.2 类方法](#32-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [3.3 类属性](#33-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [Examples](#examples)
        - [AdaBoost Example](#adaboost-example)
        - [GBDT Example](#gbdt-example)
        - [Random Forest Example](#random-forest-example)
    - [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 一、AdaBoost算法

```python
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import ClassifierType


class AdaBoost(BaseClassifier):

    __doc__ = "AdaBoost Classifier"

    def __init__(self, classifier=ClassifierType.LR, nums=10):
        pass
```


`AdaBoost`模型，为Boosting模型中的一种，支持：
- 二分类问题

子分类器支持：
- logistic分类器
- KNN分类器
- CART分类器
- SVM分类器
- NaiveBayes分类器

* * *

### 1.1 初始化

|             |       名称       |                  类型                   |           描述           |
|------------:|:----------------:|:---------------------------------------:|:------------------------:|
| Parameters: |    classifier    | [ClassifiterType](../structure/enum.md) |       子分类器类型        |
|             |       nums           |       int                                  |      子分类器数目                    |


### 1.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     训练集特征      |
|             |  y   |  np.array  | 训练集标签 |
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

None


## 二、梯度提升树 (GBDT)


```python
from simple_ml.base.base_model import BaseClassifier, BaseFeatureSelect

class GBDT(BaseClassifier, BaseFeatureSelect):

    __doc__ = "GBDT Regression"

    def __init__(self, nums=10, learning_rate=1):
        pass
```

`GBDT`模型采用CART模型作为子模型，并且支持：
- 回归问题

由于`GBDT` 继承了`BaseFeatureSelect`抽象类，因此可以进行特征选择

* * *

### 2.1 初始化

|             |       名称       |                  类型                   |           描述           |
|------------:|:----------------:|:---------------------------------------:|:------------------------:|
| Parameters: |    classifier    | [ClassifiterType](../structure/enum.md) |       子分类器类型        |
|             |       nums           |       int                                  |      子分类器数目                    |


### 2.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     训练集特征      |
|             |  y   |  np.array  | 训练集标签 |
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

5 特征选择

选择预测效果最好的前top_n个特征

```python
def feature_select(self, top_n)
```


|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | top_n | int  |  需要的特征数目        |
| Returns:    |         | np.array | 选中特征的索引        |




### 2.3 类属性

|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| importance |    np.array(float)  |  特征的重要程度数组 |


## 随机森林 (RandomForest)

```python
from simple_ml.base.base_model import BaseClassifier


class RandomForest(BaseClassifier):

    __doc__ = "Random Forest"

    def __init__(self, m, tree_num=200):
        pass
```

随机森林(Random Forest)模型同样采用CART作为子分类器，采用投票的方式作为决策依据

支持：
- 二分类问题
- 多分类问题

* * *

### 3.1 初始化

|             |       名称       |                  类型                   |           描述           |
|------------:|:----------------:|:---------------------------------------:|:------------------------:|
| Parameters: |    classifier    | [ClassifiterType](../structure/enum.md) |       子分类器类型        |
|             |       nums           |       int                                  |      子分类器数目                    |


### 3.2 类方法

1 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     训练集特征      |
|             |  y   |  np.array  | 训练集标签 |
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

### 3.3 类属性

|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| the_forest |   List[BinaryTreeNode](../structure/struct.md)   |  每棵树的列表 |


## Examples

### AdaBoost Example

```python
from simple_ml.ensemble import AdaBoost
from simple_ml.classify_data import get_wine
from simple_ml.data_handle import train_test_split
from simple_ml.base.base_enum import *


x, y = get_wine()
x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

# 采用Logistic回归作为子分类器的AdaBoost
ada = AdaBoost(classifier=ClassifierType.LR)
ada.fit(x_train, y_train)
print(ada.score(x_test, y_test))
ada.classify_plot(x_test, y_test, ", LR")

# 采用KNN作为子分类器的AdaBoost
ada = AdaBoost(classifier=ClassifierType.KNN)
ada.fit(x_train, y_train)
print(ada.score(x_test, y_test))
ada.classify_plot(x_test, y_test, ", KNN")

# 采用CART树为子分类器的AdaBoost
ada = AdaBoost(classifier=ClassifierType.CART)
ada.fit(x_train, y_train)
print(ada.score(x_test, y_test))
ada.classify_plot(x_test, y_test, ", CART")
```

### GBDT Example

```python
from simple_ml.ensemble import *
from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split


x, y = get_watermelon()
y = x[:, -1]     # y为连续标签
x = x[:, :-1]    # x为离散标签
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

gbdt = GBDT(learning_rate=1)
gbdt.fit(x_train, y_train)
print(gbdt.predict(x_test), y_test)
print("R square: %.4f" % gbdt.score(x_test, y_test))

x, y = get_wine()
y = x[:, -1]  # y为连续标签
x = x[:, :-1]  # x为离散标签
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

gbdt = GBDT(learning_rate=1)
gbdt.fit(x_train, y_train)
print(gbdt.predict(x_test), y_test)
print("R square: %.4f" % gbdt.score(x_test, y_test))
```


### Random Forest Example

```python
from simple_ml.classify_data import *
from simple_ml.data_handle import *
from simple_ml.ensemble import RandomForest


x, y = get_wine()
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

rf = RandomForest(4, 50)
rf.fit(x_train, y_train)
print(rf.score(x_test, y_test))
rf.classify_plot(x_test, y_test)

```

## [返回主页](../index.md)
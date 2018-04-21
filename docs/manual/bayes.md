# 贝叶斯模块 **simple_ml.bayes**

- [贝叶斯模块 **simple_ml.bayes**](#%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%A8%A1%E5%9D%97-simplemlbayes)
    - [一、朴素贝叶斯 (NaiveBayes)](#%E4%B8%80%E3%80%81%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF-naivebayes)
        - [初始化](#%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [类方法](#%E7%B1%BB%E6%96%B9%E6%B3%95)
    - [二、贝叶斯最小误差 (BME)](#%E4%BA%8C%E3%80%81%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%9C%80%E5%B0%8F%E8%AF%AF%E5%B7%AE-bme)
        - [初始化](#%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [类方法](#%E7%B1%BB%E6%96%B9%E6%B3%95)
    - [三、贝叶斯最小损失 (BMR)](#%E4%B8%89%E3%80%81%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%9C%80%E5%B0%8F%E6%8D%9F%E5%A4%B1-bmr)
        - [初始化](#%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [类方法](#%E7%B1%BB%E6%96%B9%E6%B3%95)
    - [Example](#example)
        - [Binary Example](#binary-example)
        - [Multi-class Example](#multi-class-example)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 一、朴素贝叶斯 (NaiveBayes)


```python
from simple_ml.base.base_model import BaseClassifier


class NaiveBayes(BaseClassifier):

    __doc__ = "Naive Bayes Classifier"

    def __init__(self):
        pass
```

朴素贝叶斯模型，支持解决：
- 二分类问题
- 多分类问题

* * *

### 初始化

None

### 类方法

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



**类属性**


|     名称      |    类型    |         描述          |
|:-------------:|:----------:|:---------------------:|
| posterior_prob |   np.2darray(float)   | P(X|Y)，x的后验概率，shape=(C, m+1)，C为Y取值数目，m为特征数目，第一列为Y的先验 |


## 二、贝叶斯最小误差 (BME)


```python
from simple_ml.base.base_model import BaseClassifier


class BayesMinimumError(BaseClassifier):

    __doc__ = "Bayes Minimum Error"

    def __init__(self):
        pass
```

贝叶斯最小误差模型，支持解决：
- 二分类问题
- 多分类问题

* * *

### 初始化

None

### 类方法

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



**类属性**


|  名称  |       类型        |    描述     |
|:------:|:-----------------:|:-----------:|
| sigma | np.2darray(float) | 样本协方差阵 |
| mu       | np.array(float)                  | 样本均值向量            |


## 三、贝叶斯最小损失 (BMR)


```python
from simple_ml.bayes import BayesMinimumError

class BayesMinimumRisk(BayesMinimumError):

    __doc__ = "Bayes Minimum Risk"

    def __init__(self, cost_mat):
        """
        初始化，保存分类损失矩阵
        :param cost_mat: 分类损失矩阵
               要求：
                   1. m x m 维，m为所有类别数目
                   2. 第i行第j列表示将属于类别i的样本分到类别j所造成的损失
                   3. 每一行，每一列的类别必须按照数值从小到大的顺序排列，
                      比如第i行表示在np.unique(y)中第i个label
        """
        super(BayesMinimumRisk, self).__init__()
        self.cost_mat = cost_mat
```

贝叶斯最小损失模型，支持解决：
- 二分类问题
- 多分类问题

* * *

### 初始化

|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | cost_mat | np.2darray  | 损失矩阵 ，见代码中的注释        |

### 类方法

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



**类属性**


|  名称  |       类型        |    描述     |
|:------:|:-----------------:|:-----------:|
| sigma | np.2darray(float) | 样本协方差阵 |
| mu       | np.array(float)                  | 样本均值向量            |


## Example

### Binary Example
```python
from simple_ml.classify_data import get_wine
from simple_ml.bayes import BayesMinimumError, BayesMinimumRisk, NaiveBayes
import numpy as np
from simple_ml.data_handle import train_test_split

x, y = get_wine()

x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

# 贝叶斯最小错误率
bme = BayesMinimumError()
bme.fit(x_train, y_train)
print(bme.score(x_test, y_test))
bme.classify_plot(x_test, y_test)

# 贝叶斯最小风险，需要给定风险矩阵
# 风险矩阵 [[0,100], [10,0]] 表示把0分为1（存伪）的损失为100，把1分为0（弃真）的损失为10
bmr = BayesMinimumRisk(np.array([[0, 100], [10, 0]]))
bmr.fit(x_train, y_train)
bmr.predict(x_test)
print(bmr.score(x_test, y_test))
bmr.classify_plot(x_test, y_test)

# 朴素贝叶斯
nb = NaiveBayes()
nb.fit(x_train, y_train)
nb.predict(x_test)
print(nb.score(x_test, y_test))
nb.classify_plot(x_test, y_test)
```

### Multi-class Example

```python
from simple_ml.classify_data import get_wine
from simple_ml.bayes import BayesMinimumError, BayesMinimumRisk, NaiveBayes
import numpy as np
from simple_ml.data_handle import train_test_split

x, y = get_wine()

x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

# 贝叶斯最小错误率
bme = BayesMinimumError()
bme.fit(x_train, y_train)
print(bme.score(x_test, y_test))
bme.classify_plot(x_test, y_test)

# 贝叶斯最小风险，需要给定风险矩阵
# 风险矩阵 [[0,100], [10,0]] 表示把0分为1（存伪）的损失为100，把1分为0（弃真）的损失为10
bmr = BayesMinimumRisk(np.array([[0, 100, 10], [10, 0, 100], [10, 10, 0]]))
bmr.fit(x_train, y_train)
bmr.predict(x_test)
print(bmr.score(x_test, y_test))
bmr.classify_plot(x_test, y_test)

# 朴素贝叶斯
nb = NaiveBayes()
nb.fit(x_train, y_train)
nb.predict(x_test)
print(nb.score(x_test, y_test))
nb.classify_plot(x_test, y_test)
```


# [返回主页](../index.md)
# 支持向量学习 **simple_ml.svm**

- [支持向量学习 **simple_ml.svm**](#%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E5%AD%A6%E4%B9%A0-simplemlsvm)
    - [支持向量机 (SVM)](#%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA-svm)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [Examples](#examples)
        - [线性可分情况](#%E7%BA%BF%E6%80%A7%E5%8F%AF%E5%88%86%E6%83%85%E5%86%B5)
        - [线性不可分，软间隔情况](#%E7%BA%BF%E6%80%A7%E4%B8%8D%E5%8F%AF%E5%88%86%EF%BC%8C%E8%BD%AF%E9%97%B4%E9%9A%94%E6%83%85%E5%86%B5)
        - [线性不可分，高维可分情况](#%E7%BA%BF%E6%80%A7%E4%B8%8D%E5%8F%AF%E5%88%86%EF%BC%8C%E9%AB%98%E7%BB%B4%E5%8F%AF%E5%88%86%E6%83%85%E5%86%B5)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *
## 一、支持向量机 (SVM)

```python
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import KernelType

class SVM(BaseClassifier):

    __doc__ = "SVM"

    def __init__(self, c=0.1, tol=0.001, precision=0.01, max_iter=100, kernel_type=KernelType.linear, **kwargs):
        """
        可变参数kwargs存储核函数的参数
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束）
            tol         误差容忍度（越大越不准确，但是省时间）
            precision   alpha结果精度
            max_iter    外循环最大迭代次数
            kernel_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        pass
```

`svm`支持向量机模块，支持：
- 二分类

核函数支持：
- linear 线性核
- polynomial 多项式核
- gaussian 高斯核
- laplace 拉普拉斯核
- sigmoid核


### 1.1 初始化

|             |    名称     |                类型                |        描述        |
|------------:|:-----------:|:----------------------------------:|:------------------:|
| Parameters: |      C      |               float                | 软间隔支持向量机参数 |
|             |     tol     |               float                |     误差容忍度      |
|             |  precision  |               float                |    aloha结果精度    |
|             |  max_iter   |                int                 |   外循环迭代次数    |
|             | kernel_type | [KernalType](../structure/enum.md) |     核函数类型，见注释，需要加入相应参数               |


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
|    Returns: |      |   float    | 预测结果评分，此处给出F1值 |

4 分类作图

`logistic`模块提供了直接绘制分类效果图的方法，如果维度大于2，则通过PCA降至两维

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

## Examples

### 线性可分情况

```python
from simple_ml.svm import *
from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split

x, y = get_iris()
x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test)
```

### 线性不可分，软间隔情况

```python
from simple_ml.svm import *
from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split

x, y = get_iris()
x = x[(y == 1) | (y == 2)]
y = y[(y == 1) | (y == 2)]

x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test)
```


### 线性不可分，高维可分情况

```python
from simple_ml.svm import *
from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split

x, y = get_moon()
x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test, ", Linear")

# sigma设置的比较小，会过拟合
mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.gaussian, sigma=0.5)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test, ", Gaussian(sigma=0.5)")

# sigma设置的比较大，会欠拟合
mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.gaussian, sigma=1)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test, ", Gaussian(sigma=1.0)")

mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.laplace, sigma=1)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test, ", Laplace(sigma=1)")

mysvm = SVM(0.6, 0.001, 0.00001, 50, KernelType.sigmoid, beta=1, theta=-1)
mysvm.fit(x_train, y_train)
print(mysvm.alphas, mysvm.b)
print(mysvm.predict(x_train))
mysvm.classify_plot(x_test, y_test, ", Sigmoid(beta=1,theta=1)")

```

# [返回主页](../index.md)
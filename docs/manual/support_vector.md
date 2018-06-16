# 支持向量学习 **simple_ml.support_vector**



* * *

## 一、核函数静态类

```python
class Kernel(object):
```

### 1.1 线性核(linear)

```python
@statistics
def linear():
    pass
```

### 1.2 高斯核(gaussian)

```python
@staticmethod
def gaussian(sigma):
    pass
```


### 1.3 多项式核(polykernel)

```python
@staticmethod
def polykernel(dimension, offset=0.0):
    pass
```

### 1.4 拉普拉斯核(laplace)

```python
@staticmethod
def laplace(sigma):
    pass
```

### 1.5 tanh核/sigmoid核(hyperbolic_tangent)

```python
@staticmethod
def hyperbolic_tangent(kappa, c):
    pass
```


## 二、支持向量类(BaseSupportVector)

该类主要用来添加获取相应核函数的方法

```python
class BaseSupportVector:

    @staticmethod
    def _get_kernel_func(kernel_name, kwargs):
        if kernel_name == KernelType.linear:
            return Kernel.linear()
        elif kernel_name == KernelType.gaussian:
            return Kernel.gaussian(kwargs['sigma'])
        elif kernel_name == KernelType.polynomial:
            if 'o' in kwargs:
                return Kernel.polykernel(kwargs['d'], kwargs['o'])
            else:
                return Kernel.polykernel(kwargs['d'])
        elif kernel_name == KernelType.laplace:
            return Kernel.laplace(kwargs['sigma'])
        elif kernel_name == KernelType.sigmoid:
            return Kernel.hyperbolic_tangent(kwargs['beta'], kwargs['theta'])
        else:
            raise KernelTypeError("非法的核函数名称")
```

## 三、支持向量机 (PD优化包求解)

```python
class SVR(BaseSupportVector, BaseClassifier):

    __doc__ = "Support Vector Regression"

    def __init__(self, c, eps=0.1, kernel=KernelType.linear, **kwargs):
        """
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束）
            eps         容忍的带宽
            kernel_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        super(SVR, self).__init__()
```

`SVR`支持向量回归模块，支持：
- 线性回归
- 非线性回归

核函数支持：
- linear 线性核
- polynomial 多项式核
- gaussian 高斯核
- laplace 拉普拉斯核
- sigmoid核


### 3.1 初始化

|             |    名称     |                类型                |        描述        |
|------------:|:-----------:|:----------------------------------:|:------------------:|
| Parameters: |      C      |               float                | 软间隔支持向量机参数 |
|             |  eps |                float               |   可以容忍的带宽宽度 |
|             | kernel_type | [KernalType](../structure/enum.md) |     核函数类型，见注释，需要加入相应参数               |


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
|    Returns: |      |   float    | 预测结果评分，此处给出F1值 |

4 回归作图


```python
def regression_plot(self, x):
    y = self.predict(x)
    regression_plot(self.x, self.y, x, y, title="SVR")
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|    Returns: |      |    Void    |           |

### 3.3 类属性

| 名称              | 类型         | 描述                    |
| :-------------:   | :----------: | :---------------------: |
| weight            | np.array     | 每个特征的参数          |
| bias              | float        | 偏移项                  |
| support_vector_id | np.array     | 支持向量样本的id        |


## 四、支持向量机(SVM)


```python
class SVM(BaseClassifier, BaseSupportVector, Multi2Binary):

    __doc__ = "Support Vector Machine"

    def __init__(self, c, eps=0.1, kernel=KernelType.linear, **kwargs):
        """
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束）
            eps         容忍的带宽
            kernel_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        super(SVM, self).__init__()
```

`SVM`支持向量机，支持：
- 二分类
- 多酚类

核函数支持：
- linear 线性核
- polynomial 多项式核
- gaussian 高斯核
- laplace 拉普拉斯核
- sigmoid核


### 4.1 初始化

|             |    名称     |                类型                |        描述        |
|------------:|:-----------:|:----------------------------------:|:------------------:|
| Parameters: |      C      |               float                | 软间隔支持向量机参数 |
|             |  eps |                float               |   可以容忍的带宽宽度 |
|             | kernel_type | [KernalType](../structure/enum.md) |     核函数类型，见注释，需要加入相应参数               |


### 4.2 类方法

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


```python
def classify_plot(self, x, y, title=""):
        classify_plot(self.new(), self.x, self.y, x, y, title=self.__doc__ + title)
```

|             | 名称 | 类型       | 描述       |
|-------------|------|------------|------------|
| Parameters: | x    | np.2darray | 测试集特征 |
|             | y    | np.2darray | 测试集标签 |
| Returns:    |      | Void       |            |


### 4.3 类属性

| 名称              | 类型         | 描述                    |
| :-------------:   | :----------: | :---------------------: |
| weight            | np.array     | 每个特征的参数          |
| bias              | float        | 偏移项                  |
| support_vector_id | np.array     | 支持向量样本的id        |



## Examples

# [返回主页](../index.md)

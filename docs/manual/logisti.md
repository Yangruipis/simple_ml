学习模块 **simple_ml.knn**


## 标准Logistic回归


```python
from simple_ml.base.base_model import BaseClassifier

class LogisticRegression(BaseClassifier):

    __doc__ = "Logistic Regression"

    def __init__(self, tol=0.01, alpha=0.01, threshold=0.5, 
                has_intercept=True, sample_weights=None):
        """
        不包含惩罚项的Logistic回归
        :param tol:            误差容忍度，越大时收敛越快，但是越不精确
        :param alpha:           步长，梯度下降的参数
        :param threshold:      决策阈值，当得到的概率大于等于阈值时输出1，否则输出0
        :param has_intercept:  是否包含截距项
        """
        pass
```

`simple_ml`
提供了标准的Logistic回归模型，采用梯度下降进行参数估计，后面会加入更多优化方法

Logistic回归支持：
- 二分类问题


* * *

### 初始化

|             |     名称      |     类型     |              描述               |
|------------:|:-------------:|:------------:|:-------------------------------:|
| Parameters: |      tol      |    float     |            误差容忍度            |
|             |     step      | float, (0,1] |           梯度下降步长           |
|             |   threshold   | float,(0,1)  | 分类阈值，大于该值为1，小于该值为0 |
|             | has_intercept |     bool     |          是否含有截距项          |


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

5 概率预测

```python
def predict_prob(self, x)
```

给定测试集特征x，对其正负类的概率进行预测

|             | 名称 |       类型       |             描述              |
|------------:|:----:|:---------------:|:----------------------------:|
| Parameters: |  x   |   np.2darray    |          测试集特征           |
|    Returns: |      | np.array(float) | 预测的结果，表示为正类(1)的概率 |

6 ROC曲线绘制

针对二分类且可以包含
`predict_prob`方法的模型，我们均给出了ROC曲线的绘制方法`auc_plot`，并且在图中输出AUC值

```python
def auc_plot(self, x, y)
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |


**类属性**

None

## 最小损失回归 (Lasso)



```python
from simple_ml.base.base_model import BaseFeatureSelect
from simple_ml.logistic import LogisticRegression

class Lasso(LogisticRegression, BaseFeatureSelect):

    __doc__ = "Lasso Regression"

    def __init__(self, tol=0.01, lamb=0.1, alpha=0.01, threshold=0.5, has_intercept=True):
        """
        包含L1惩罚项的Logistic回归
        :param tol:            误差容忍度，越大时收敛越快，但是越不精确
        :param lamb            lambda，即惩罚项前面的参数，越大越不容易过拟合，但是偏差也越大
        :param alpha:           步长，梯度下降的参数
        :param threshold:      决策阈值，当得到的概率大于等于阈值时输出1，否则输出0
        :param has_intercept:  是否包含截距项
        """
        super(Lasso, self).__init__(tol, alpha, threshold, has_intercept)
        self.lamb = lamb
```

`simple_ml` 提供了带`L1正则项`的Logistic回归模型，即Lasso模型，采用**坐标下降**进行参数估计，后面会加入更多优化方法

Logistic回归支持：
- 二分类问题


* * *

### 初始化

|             |     名称      |     类型     |              描述               |
|------------:|:-------------:|:------------:|:-------------------------------:|
| Parameters: |      tol      |    float     |            误差容忍度            |
|             |     step      | float, (0,1] |           梯度下降步长           |
|             |   threshold   | float,(0,1)  | 分类阈值，大于该值为1，小于该值为0 |
|             | has_intercept |     bool     |          是否含有截距项          |


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

5 概率预测

```python
def predict_prob(self, x)
```

给定测试集特征x，对其正负类的概率进行预测

|             | 名称 |       类型       |             描述              |
|------------:|:----:|:---------------:|:----------------------------:|
| Parameters: |  x   |   np.2darray    |          测试集特征           |
|    Returns: |      | np.array(float) | 预测的结果，表示为正类(1)的概率 |

6 ROC曲线绘制

针对二分类且可以包含
`predict_prob`方法的模型，我们均给出了ROC曲线的绘制方法`auc_plot`，并且在图中输出AUC值

```python
def auc_plot(self, x, y)
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |


**类属性**

None

## 岭回归 (Ridge)

```python
from simple_ml.base.base_model import BaseFeatureSelect
from simple_ml.logistic import LogisticRegression

class Lasso(LogisticRegression, BaseFeatureSelect):

    __doc__ = "Lasso Regression"

    def __init__(self, tol=0.01, lamb=0.1, alpha=0.01, threshold=0.5, has_intercept=True):
        """
        包含L1惩罚项的Logistic回归
        :param tol:            误差容忍度，越大时收敛越快，但是越不精确
        :param lamb            lambda，即惩罚项前面的参数，越大越不容易过拟合，但是偏差也越大
        :param alpha:           步长，梯度下降的参数
        :param threshold:      决策阈值，当得到的概率大于等于阈值时输出1，否则输出0
        :param has_intercept:  是否包含截距项
        """
        super(Lasso, self).__init__(tol, alpha, threshold, has_intercept)
        self.lamb = lamb
```

`simple_ml` 提供了带`L1正则项`的Logistic回归模型，即Lasso模型，采用**坐标下降**进行参数估计，后面会加入更多优化方法

同时，

Logistic回归支持：
- 二分类问题


* * *

### 初始化

|             |     名称      |     类型     |              描述               |
|------------:|:-------------:|:------------:|:-------------------------------:|
| Parameters: |      tol      |    float     |            误差容忍度            |
|             |     step      | float, (0,1] |           梯度下降步长           |
|             |   threshold   | float,(0,1)  | 分类阈值，大于该值为1，小于该值为0 |
|             | has_intercept |     bool     |          是否含有截距项          |


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

5 概率预测

```python
def predict_prob(self, x)
```

给定测试集特征x，对其正负类的概率进行预测

|             | 名称 |       类型       |             描述              |
|------------:|:----:|:---------------:|:----------------------------:|
| Parameters: |  x   |   np.2darray    |          测试集特征           |
|    Returns: |      | np.array(float) | 预测的结果，表示为正类(1)的概率 |

6 ROC曲线绘制

针对二分类且可以包含 `predict_prob`方法的模型，我们均给出了ROC曲线的绘制方法`auc_plot`，并且在图中输出AUC值

```python
def auc_plot(self, x, y)
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |


**类属性**

None

# [返回主页](../index.md)


# 神经网络模块 **simple_ml.neural_network**

- [神经网络模块 **simple_ml.neural_network**](#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9D%97-simplemlneuralnetwork)
    - [反向传播网络 (BP Network)](#%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%BD%91%E7%BB%9C-bp-network)
        - [1.1 初始化](#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
        - [1.2 类方法](#12-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [1.3 类属性](#13-%E7%B1%BB%E5%B1%9E%E6%80%A7)
    - [Examples](#examples)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 反向传播网络 (BP Network)

```python
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import ActiveFunction, CostFunction


class NeuralNetwork(BaseClassifier):

    __doc__ = "BP Neural Network"
    
    def __init__(self, alpha=0.5, threshold=0.5, iter_times=100, 
            output_neuron_num=1, output_active_func=ActiveFunction.sigmoid, 
            cost_func=CostFunction.logistic):
        pass
```

`NeuralNetwork`模型通过反向传播(BP)算法估计参数，可以随意添加隐含层

当前支持：
- 二分类问题

神经元激活函数支持：
- sigmoid函数
- tanh函数
- ReLu函数

神经网络损失函数支持：
- logistic损失
- 平方损失

* * *

### 1.1 初始化

|             |        名称        |                  类型                  |               描述               |
|------------:|:------------------:|:--------------------------------------:|:--------------------------------:|
| Parameters: |       alpha        |                 float                  |         梯度下降更新步长          |
|             |     threshold      |              float, (0,1)              | 分类阈值，大于等于该值为1，否则为0  |
|             |     iter_times     |                  int                   |         反向传播迭代次数          |
|             | output_neuron_num  |                  int                   | 输出层神经元个数，如果是二分类就是1 |
|             | output_active_func | [ActiveFunction](../structure/enum.md) |         输出层激活函数类型         |
|             |     cost_func      |   [cost_func](../structure/enum.md)    |        神经网络损失函数类型        |


### 1.2 类方法


1 添加隐含层

```python
def add_layer(self, neuron_num, active_func=ActiveFunction.sigmoid)
```

|             |    名称     |                  类型                  |     描述      |
|------------:|:-----------:|:--------------------------------------:|:-------------:|
| Parameters: | neuron_num  |                  int                   | 该层神经元个数 |
|             | active_func | [ActiveFunction](../structure/enum.md) |  激活函数类型  |
|    Returns: |    Void     |                                        |               |

2 批量添加隐含层

```python
def add_some_layers(self, layer_num, neuron_num, active_func=ActiveFunction.sigmoid)
```

|             |    名称     |                  类型                  |     描述      |
|------------:|:-----------:|:--------------------------------------:|:-------------:|
| Parameters: | neuron_num  |                  int                   | 该层神经元个数 |
|             |      layer_num       |      int                                  |   添加隐含层层数            |
|             | active_func | [ActiveFunction](../structure/enum.md) |  激活函数类型  |
|    Returns: |    Void     |                                        |               |

3 清空隐含层

```python
def clear_all(self)
```

4 拟合

```python
def fit(self, x, y)
```

拟合特征

|             | 名称 |    类型     |     描述      |
|------------:|:----:|:----------:|:------------:|
| Parameters: |  x   | np.2darray |     训练集特征      |
|             |  y   |  np.array  | 训练集标签 |
|    Returns: |      |    Void    |              |


5 预测

```python
def predict(self, x)
```


给定测试集特征x，进行预测

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|    Returns: |      |  np.array  | 预测的结果 |

6 结果评价

```python
def score(self, x, y)
```

拟合并进行预测，最后给出预测效果的得分


|             | 名称 |    类型     |                            描述                            |
|------------:|:----:|:----------:|:---------------------------------------------------------:|
| Parameters: |  x   | np.2darray |                         测试集特征                         |
|             |  y   |  np.array  |                         测试集标签                         |
|    Returns: |      |   float    | 预测结果评分，此处给出F1值 |

7 分类作图

`logistic`模块提供了直接绘制分类效果图的方法，如果维度大于2，则通过PCA降至两维

```python
def classify_plot(self, x, y, title="")
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |
|    Returns: |      |    Void    |           |

8 概率预测

```python
def predict_prob(self, x)
```

给定测试集特征x，对其正负类的概率进行预测

|             | 名称 |       类型       |             描述              |
|------------:|:----:|:---------------:|:----------------------------:|
| Parameters: |  x   |   np.2darray    |          测试集特征           |
|    Returns: |      | np.array(float) | 预测的结果，表示为正类(1)的概率 |

9 ROC曲线绘制

针对二分类且包含
`predict_prob`方法的模型，我们均给出了ROC曲线的绘制方法`auc_plot`，并且在图中输出AUC值

```python
def auc_plot(self, x, y)
```

|             | 名称 |    类型     |    描述    |
|------------:|:----:|:----------:|:---------:|
| Parameters: |  x   | np.2darray | 测试集特征 |
|             |  y   |  np.array  | 测试集标签 |

### 1.3 类属性

None

## Examples

```python
from simple_ml.classify_data import *
from simple_ml.neural_network import *
from simple_ml.data_handle import train_test_split
from simple_ml.base.base_enum import *


x, y = get_wine()

x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 918)

nn = NeuralNetwork(alpha=0.5, cost_func=CostFunction.square)
nn.add_some_layers(2, 3, active_func=ActiveFunction.relu)
nn.fit(x_train, y_train)
print(nn.predict_prob(x_test))
nn.classify_plot(x_test, y_test)
nn.auc_plot(x_test, y_test)
```

# [返回主页](../index.md)
# 模型抽象类 simple_ml.base.base_model


- [模型抽象类 simple_ml.base.base_model](#%E6%A8%A1%E5%9E%8B%E6%8A%BD%E8%B1%A1%E7%B1%BB-simplemlbasebasemodel)
    - [一、分类模型](#%E4%B8%80%E3%80%81%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B)
        - [1.1 继承关系](#11-%E7%BB%A7%E6%89%BF%E5%85%B3%E7%B3%BB)
        - [1.2 方法](#12-%E6%96%B9%E6%B3%95)
        - [3. 成员属性](#3-%E6%88%90%E5%91%98%E5%B1%9E%E6%80%A7)
    - [二、转换模型](#%E4%BA%8C%E3%80%81%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B)
        - [2.1 继承关系](#21-%E7%BB%A7%E6%89%BF%E5%85%B3%E7%B3%BB)
        - [2.2 方法](#22-%E6%96%B9%E6%B3%95)
        - [2.3 成员属性](#23-%E6%88%90%E5%91%98%E5%B1%9E%E6%80%A7)
    - [三、多分类转二分类 (Multi2Binary)](#%E4%B8%89%E3%80%81%E5%A4%9A%E5%88%86%E7%B1%BB%E8%BD%AC%E4%BA%8C%E5%88%86%E7%B1%BB-multi2binary)
        - [3.1 继承关系](#31-%E7%BB%A7%E6%89%BF%E5%85%B3%E7%B3%BB)
        - [3.2 类方法](#32-%E7%B1%BB%E6%96%B9%E6%B3%95)
        - [3.3 成员属性](#33-%E6%88%90%E5%91%98%E5%B1%9E%E6%80%A7)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *


## 一、分类模型

```python
class BaseClassifier(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
```

### 1.1 继承关系

**子类**

- simple_ml.bayes
  - class NaiveBayes(BaseClassifier):
  - class BaseBayesMinimumError(BaseClassifier):
  - class BayesMinimumRisk(BaseBayesMinimumError):
- simple_ml.bp_network
  - class BaseBPNetwork(BaseClassifier):
- simple_ml.ensemble
  - class AdaBoost(BaseClassifier):
  - class CARTForGBDT(CART):
  - class GBDT(BaseClassifier, BaseFeatureSelect):
- simple_ml.knn
  - class KNN(BaseClassifier):
  - class KDTree(KNN):
- simple_ml.logistic
  - class LogisticRegression(BaseClassifier):
  - class Lasso(LogisticRegression, BaseFeatureSelect):
  - class Ridge(Lasso):
- simple_ml.svm
  - class SVM(BaseClassifier):
- simple_ml.tree
  - class ID3(BaseClassifier):
  - class CART(BaseClassifier):
  - class RandomForest(BaseClassifier):

**父类**

None

### 1.2 方法

```python
def _clear()
```

清空所有变量


* * *

```python
@staticmethod
def _check_y(y)
@staticmethod
def _check_x(x)
```

检查y和x的`shape`、`变量类型` 是否满足要求

* * *

```python
@staticmethod
def _check_label_type(y)
@staticmethod
def _check_feature_type(x)
```

检查特征和标签的类别，即 `LabelType`

* * *

```python
@abstractmethod
def fit(self, x, y):
    pass

@abstractmethod
def predict(self, x):
    pass

@abstractmethod
def score(self, x, y):
    pass
```

分类模型的核心方法，子类必须重写

### 3. 成员属性

|      属性名称      |      类型      |  含义   |
|:------------------|:---------------|:--------|
| self.x            | np.2darray     | 特征    |
| self.y            | np.array       | 标签    |
| slef.sample_num   | int            | 样本数   |
| self.variable_num | int            | 特征数   |
| self.label_type   | LabelType      | 标签类别 |
| self.feature_type | List[LabelType]| 特征类别 |


## 二、转换模型

```python
class BaseTransform(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass    
```

### 2.1 继承关系

**子类**

- simple_ml.feature_select
    - class Embedded(BaseTransform):
    - class Filter(BaseTransform):
- simple_ml.pca.py  (3 usages found)
    - class PCA(BaseTransform):

**父类**

None

### 2.2 方法

```python
def _clear()
```

清空所有变量


* * *

```python
@staticmethod
def _check_y(y)
@staticmethod
def _check_x(x)
```

检查y和x的`shape`、`变量类型` 是否满足要求

* * *

```python
@staticmethod
def _check_label_type(y)
@staticmethod
def _check_feature_type(x)
```

检查特征和标签的类别，即 `LabelType`

* * *

```python
@abstractmethod
def fit(self, x, y):
    pass

@abstractmethod
def transform(self, x):
    pass

@abstractmethod
def fit_transform(self, x, y):
    pass
```

转换模型的核心方法，子类必须重写

### 2.3 成员属性

|      属性名称      |      类型      |  含义   |
|:------------------|:---------------|:--------|
| self.x            | np.2darray     | 特征    |
| self.y            | np.array       | 标签    |
| slef.sample_num   | int            | 样本数   |
| self.variable_num | int            | 特征数   |
| self.label_type   | LabelType      | 标签类别 |
| self.feature_type | List[LabelType]| 特征类别 |

## 三、多分类转二分类 (Multi2Binary)

```python
class Multi2Binary:

    def __init__(self):
        pass
```

针对部分模型难以从模型角度解决多分类问题，比如SVM， Logistic，AdaBoost，
`simple_ml`构建了一个转换类，通过实例化多个分类器，来进行多分类，具体做法如下：

- 对标签[0,1,1,2]进行分类
- 构建三个二元分类器，如Logistic回归
- 每个分类器分别对 [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]三个标签进行拟合
- 针对预测样本，举个例子：
  - 对样本x， 分类器1预测结果为1， 分类器2预测结果为0， 分类器3预测结果为1
  - 那么，分类器1给出结果为类“0”，分类器2给出结果为“类0或类2”，分类器3给出结果为“类2”
  - 计数得到：{类0:2, 类1:0, 类2:2}，根据计数大小，选择最大且靠前（靠前仅仅是为了统一结果)的类别

### 3.1 继承关系


- simple_ml.ensemble
  - class AdaBoost(BaseClassifier, Multi2Binary)
- simple_ml.logistic
  - class LogisticRegression(BaseClassifier, Multi2Binary)
  - class Lasso(LogisticRegression)
  - class Ridge(Lasso)
- simple_ml.neural_network
  - class NeuralNetwork(BaseClassifier, Multi2Binary)
- simple_ml.svm
  - class SVM(BaseClassifier, Multi2Binary)


### 3.2 类方法

1 多值拟合

```python
def _multi_fit(self, model)
```

当子类标签类别为多值时，在子类的fit函数中调用，传入实例本身：

`self._multi_fit(self)`

|             | 名称  |      类型      |   描述    |
|------------:|:-----:|:--------------:|:---------:|
| Parameters: | model | BaseClassifier | 二元分类器 |
|    Returns: |       |      Void      |           |

2 多值单样本预测

```python
def _multi_predict_single(self, x)
```

|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | x | np.array  |  单样本测试集特征        |
| Returns:    |         | np.array |  预测标签       |


3 多值多样本预测

```python
 def _multi_predict(self, x)
```

|             |  名称   |  类型    |  描述           |
| ----------: | :-----: |:-------:| :--------------:|
| Parameters: | x | np.2darray  |  测试集特征        |
| Returns:    |         | np.array |  预测标签       |


### 3.3 成员属性

| 属性名称    | 类型                 | 含义                      |
|:-----------|:---------------------|:--------------------------|
| new_models | List[BaseClassifier] | 分类器列表，长度等于标签类别 |
| model_num  | int                  | 分类器数目                 |
| y_unique   | np.array             | 标签唯一值                 |




# [返回主页](../index.md)


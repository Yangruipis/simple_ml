# 模型抽象类 simple_ml.base.base_model


- [模型抽象类 simple_ml.base.base_model](#%E6%A8%A1%E5%9E%8B%E6%8A%BD%E8%B1%A1%E7%B1%BB-simplemlbasebasemodel)
    - [分类模型抽象类](#%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E6%8A%BD%E8%B1%A1%E7%B1%BB)
        - [继承关系](#%E7%BB%A7%E6%89%BF%E5%85%B3%E7%B3%BB)
        - [方法](#%E6%96%B9%E6%B3%95)
        - [属性](#%E5%B1%9E%E6%80%A7)
    - [转换模型抽象类](#%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B%E6%8A%BD%E8%B1%A1%E7%B1%BB)
        - [继承关系](#%E7%BB%A7%E6%89%BF%E5%85%B3%E7%B3%BB)
        - [方法](#%E6%96%B9%E6%B3%95)
        - [属性](#%E5%B1%9E%E6%80%A7)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *


## 分类模型抽象类

```python
class BaseClassifier(object):
    
    __metaclass__ = ABCMeta    
```

### 继承关系

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

### 方法

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

### 属性

|      属性名称      |      类型      |  含义   |
|:------------------|:---------------|:--------|
| self.x            | np.2darray     | 特征    |
| self.y            | np.array       | 标签    |
| slef.sample_num   | int            | 样本数   |
| self.variable_num | int            | 特征数   |
| self.label_type   | LabelType      | 标签类别 |
| self.feature_type | List[LabelType]| 特征类别 |


## 转换模型抽象类

```python
class BaseTransform(object):
    
    __metaclass__ = ABCMeta    
```

### 继承关系

**子类**

- simple_ml.feature_select
    - class Embedded(BaseTransform):
    - class Filter(BaseTransform):
- simple_ml.pca.py  (3 usages found)
    - class PCA(BaseTransform):

**父类**

None

### 方法

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

### 属性

|      属性名称      |      类型      |  含义   |
|:------------------|:---------------|:--------|
| self.x            | np.2darray     | 特征    |
| self.y            | np.array       | 标签    |
| slef.sample_num   | int            | 样本数   |
| self.variable_num | int            | 特征数   |
| self.label_type   | LabelType      | 标签类别 |
| self.feature_type | List[LabelType]| 特征类别 |


# [返回主页](../index.md)


# 模型评价模块 **simple_ml.evaluation**

- [模型评价模块 **simple_ml.evaluation**](#%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7%E6%A8%A1%E5%9D%97-simplemlevaluation)
    - [一、二分类模型评价](#%E4%B8%80%E3%80%81%E4%BA%8C%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7)
        - [1.0 分类作图](#10-%E5%88%86%E7%B1%BB%E4%BD%9C%E5%9B%BE)
        - [1.1  classify_accuracy](#11-classifyaccuracy)
        - [1.2 classify_precision](#12-classifyprecision)
        - [1.3 classify_recall](#13-classifyrecall)
        - [1.5 classify_f1](#15-classifyf1)
        - [1.6 classify_roc](#16-classifyroc)
        - [1.7 classify_auc](#17-classifyauc)
        - [1.8 classify_roc_plot](#18-classifyrocplot)
    - [二、多分类模型评价](#%E4%BA%8C%E3%80%81%E5%A4%9A%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7)
        - [2.1 classify_f1_micro](#21-classifyf1micro)
        - [2.2 classify_f1_macro](#22-classifyf1macro)
        - [2.3 classify_f1_weighted](#23-classifyf1weighted)
    - [三、回归模型评价](#%E4%B8%89%E3%80%81%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7)
        - [3.1 regression_explained_variance](#31-regressionexplainedvariance)
        - [3.2 regression_absolute_error](#32-regressionabsoluteerror)
        - [3.3 regression_squared_error](#33-regressionsquarederror)
        - [3.4 regression_rmse](#34-regressionrmse)
        - [3.5 regression_rmsel](#35-regressionrmsel)
        - [3.6 regression_r2](#36-regressionr2)
        - [3.7 regression_median_absolute_error](#37-regressionmedianabsoluteerror)
    - [四、交叉验证](#%E5%9B%9B%E3%80%81%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

## 一、二分类模型评价

### 1.0 分类作图

```python
def classify_plot(model, x_train, y_train, x_test, y_test, 
                    title="",compare=False, px=100)
```

|             |  名称   |      类型      |                                             描述                                             |
|------------:|:-------:|:--------------:|:--------------------------------------------------------------------------------------------:|
| Parameters: |  model  | BaseClassifier | 必须是继承BaseClassifier的模型，并且必须是一个新的实例，可以通过classmethod方法传入 ，见 SVM.new() |
|             | x_train |   np.2darray   |                                          训练集特征                                           |
|             | y_train |    np.array    |                                          训练集标签                                           |
|             | x_test  |   np.2darray   |                                          测试集特征                                           |
|             | y_test  |    np.array    |                                          测试集标签                                           |
|             |  title  |      str       |                                          图的title                                           |
|             | compare |      bool      |                                   是否和原数据集进行比较作图                                   |
|             |   px    |      int       |                                           作图像素                                            |
|    Returns: |         |      Void      |                                                                                              |

### 1.1  classify_accuracy

```python
def classify_accuracy(y_predict, y_true)
```

模型准确度 = (TP + FP) / (TP + FP + TN + FN)

### 1.2 classify_precision

```python
def classify_precision(y_predict, y_true)
```

模型精度 = TP / (TP + FP)

### 1.3 classify_recall

```python
def classify_recall(y_predict, y_true)
```

模型召回率 = TP / (TP + FN)

### 1.5 classify_f1

```python
def classify_f1(y_predict, y_true)
```

F1 = 2 * precision * recall / (recall + precision)

### 1.6 classify_roc

```python
def classify_roc(y_predict, y_true)
```

获得TPR， FPR

**注意：** y_predict 必须为float数组

### 1.7 classify_auc

```python
classify_auc(y_predict, y_true)
```

根据TPR、FPR计算预测的AUC值


### 1.8 classify_roc_plot

```python
def classify_roc_plot(y_predict, y_true)
```

绘制ROC曲线


## 二、多分类模型评价

### 2.1 classify_f1_micro

```python
def classify_f1_micro(y_predict, y_true)
```

在求recall和precision时，在计算前进行加总（recall = sum(tp_i) / (sum(tp_i+fn_i))

### 2.2 classify_f1_macro

```python
def classify_f1_macro(y_predict, y_true)
```

求每一个类的recall和precision以及f1，然后求平均

### 2.3 classify_f1_weighted

```python
def classify_f1_weighted(y_predict, y_true)
```

用样本中正例数目加权（避免了样本不平衡的问题）


## 三、回归模型评价

### 3.1 regression_explained_variance

```python
def regression_explained_variance(y_predict, y_true)
```

解释平方和 (SSR)

### 3.2 regression_absolute_error

```python
def regression_absolute_error(y_predict, y_true):
```

绝对误差

### 3.3 regression_squared_error

```python
def regression_squared_error(y_predict, y_true)
```

平方误差

### 3.4 regression_rmse

```python
def regression_rmse(y_predict, y_true)
```

均方根误差

### 3.5 regression_rmsel

```python
def regression_rmsel(y_predict, y_true)
```

对数均方根误差


### 3.6 regression_r2

```python
def regression_r2(y_predict, y_true)
```

r方 = SSR / SST

### 3.7 regression_median_absolute_error

```python
def regression_median_absolute_error(y_predict, y_true)
```

中位数绝对误差

## 四、交叉验证

```python
from simple_ml.base.base_enum import CrossValidationType

def cross_validation(model, x, y, method=CrossValidationType.holdout, 
                     test_size=0.3, cv=5)
```

|             |   名称    |                    类型                     |                  描述                  |
|------------:|:---------:|:-------------------------------------------:|:--------------------------------------:|
| Parameters: |   model   |               BaseClassifier                |     必须是继承BaseClassifier的分类器     |
|             |     x     |                 np.2darray                  |                  特征                  |
|             |     y     |                  np.array                   |                  标签                  |
|             |  method   | [CrossValidationType](../structure/enum.md) |     交叉验证类型，支持hold-out和K折      |
|             | test_size |                    float                    |             测试集占的比重              |
|             |    cv     |                     int                     |   交叉验证次数，如果是K折则等同于K的值    |
|    Returns: |           |                  np.array                   | 长度等于cv的数组，表示每一次交叉验证的结果 |


# [返回主页](../index.md)
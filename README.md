<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. 特征工程</a>
<ul>
<li><a href="#sec-1-1">1.1. 特征预处理</a></li>
<ul>
<li><a href="#sec-1-1-1">1.1.1. PCA降维</a></li>
<li><a href="#sec-1-1-2">1.1.2. PCA高维降维</a></li>
</ul>
<li><a href="#sec-1-2">1.2. 特征选择</a>
<ul>
<li><a href="#sec-1-2-1">1.2.1. Filter方法</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#sec-2">2. 模型评价</a>
<ul>
<li><a href="#sec-2-1">2.1. 相关得分</a>
<ul>
<li><a href="#sec-2-1-1">2.1.1. 针对二分类问题：</a></li>
<li><a href="#sec-2-1-2">2.1.2. 针对多分类问题：</a></li>
<li><a href="#sec-2-1-3">2.1.3. 针对回归问题：</a></li>
</ul>
</li>
<li><a href="#sec-2-2">2.2. 分类结果作图</a></li>
<li><a href="#sec-2-3">2.3. 交叉验证</a></li>
</ul>
</li>
<li><a href="#sec-3">3. 分类算法</a>
<ul>
<li><a href="#sec-3-1">3.1. 类规范</a></li>
<li><a href="#sec-3-2">3.2. knn相关算法</a>
<ul>
<li><a href="#sec-3-2-1">3.2.1. 简单knn</a></li>
<li><a href="#sec-3-2-2">3.2.2. KD树</a></li>
</ul>
</li>
<li><a href="#sec-3-3">3.3. Logistic回归</a></li>
<li><a href="#sec-3-4">3.4. 贝叶斯相关算法</a>
<ul>
<li><a href="#sec-3-4-1">3.4.1. 朴素贝叶斯</a></li>
<li><a href="#sec-3-4-2">3.4.2. 半朴素贝叶斯</a></li>
<li><a href="#sec-3-4-2">3.4.3. 贝叶斯最小误差</a></li>
<li><a href="#sec-3-4-2">3.4.4. 贝叶斯最小风险</a></li>
</ul>
</li>
<li><a href="#sec-3-5">3.5. 基于树的算法</a>
<ul>
<li><a href="#sec-3-5-1">3.5.1. CART</a></li>
<li><a href="#sec-3-5-2">3.5.2. 随机森林</a></li>
</ul>
</li>
<li><a href="#sec-3-6">3.6. 支持向量机</a></li>
<li><a href="#sec-3-7">3.7. 神经网络</a>
<ul>
<li><a href="#sec-3-7-1">3.7.1. BP神经网络</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#sec-4">4. 聚类</a>
<ul>
<li><a href="#sec-4-1">4.1. K均值聚类</a></li>
<li><a href="#sec-4-2">4.2. 层次聚类</a></li>
</ul>
</li>
<li><a href="#sec-5">5. Boosting学习 </a>
<ul>
<li><a href="#sec-5-1">5.1. AdaBoost</a></li>
<li><a href="#sec-5-2">5.2. GBDT</a></li>
</ul>
</li>
</ul>
</div>
</div>

将机器学习的基本流程与算法进行手写实现，仅调用numpy以及python基本库

![](https://img.shields.io/npm/l/express.svg)  [![Codacy Badge](https://api.codacy.com/project/badge/Grade/00c639db60364d12b0102456552fe806)](https://www.codacy.com/app/Yangruipis/simpleML?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Yangruipis/simpleML&amp;utm_campaign=Badge_Grade) [![Join the chat at https://gitter.im/simple_ml/Lobby](https://badges.gitter.im/simple_ml/Lobby.svg)](https://gitter.im/simple_ml/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

TODO list:

- [ ] test cases
- [ ] an efficient bp network 
- [ ] more optimal methods
- [x] train test split func in helper
- [x] other feature select method to add
- [x] lasso and Ridge
- [x] add GBDT feature select
- [ ] update Readme
- [ ] setup.py
- [ ] examples
- [x] get more datasets
- [ ] regression plot
- [ ] kd_tree
- [ ] weighted logistic

# 特征工程<a id="sec-1" name="sec-1"></a>

## 特征预处理<a id="sec-1-1" name="sec-1-1"></a>

### PCA降维<a id="sec-1-1-1" name="sec-1-1-1"></a>
当特征数小于样本数时：
```python
from simple_ml.pca import *

pca = PCA(1)
a = np.array([[1,3,2], [3,5,1], [4,7,3], [1,2,0], [0,2,1]])
print(pca.fit_transform(a))
print(pca.explain_ratio)
```

### PCA高维降维<a id="sec-1-1-2" name="sec-1-1-2"></a>
当特征数远小于样本数时，通过矩阵分解进行低维PCA
```python
from simple_ml.pca import *

pca = SuperPCA(1)
a = np.array([[1,3,2], [3,5,1], [4,7,3], [1,2,0], [0,2,1]])
print(pca.fit_transform(a.T))
print(pca.explain_ratio)
```

## 特征选择<a id="sec-1-2" name="sec-1-2"></a>

### Filter方法<a id="sec-1-2-1" name="sec-1-2-1"></a>

当前提供了四种Filter选择方法：

-   方差法
-   相关系数法
-   卡方检验法
-   互信息法

范例如下

```python
    from simple_ml.filter_select import *
    X = np.random.random(20).reshape(-1, 4)
    Y = np.random.randint(0,2,5)
    mf = MyFilter(filter_type=FilterType.chi2, top_k=3)
    mf.fitTransform(X,Y)
    mf.transform(X)
```

# 模型评价<a id="sec-2" name="sec-2"></a>

## 相关得分<a id="sec-2-1" name="sec-2-1"></a>

### 针对二分类问题：<a id="sec-2-1-1" name="sec-2-1-1"></a>

-   accuracy
-   precision
-   recall
-   f1
-   auc
-   roc作图

### 针对多分类问题：<a id="sec-2-1-2" name="sec-2-1-2"></a>

-   f1<sub>micro</sub>
-   f1<sub>macro</sub>
-   f1<sub>weight</sub>

### 针对回归问题：<a id="sec-2-1-3" name="sec-2-1-3"></a>

-   explained<sub>variance</sub>
-   absolute<sub>error</sub>
-   squared<sub>error</sub>
-   RMSE(root mean squared error)
-   RMSLE(root mean squared log error, in case of the abnormal value)
-   r2
-   median<sub>absolute</sub><sub>error</sub>

范例：
```python
    from simple_ml.score import *
    print(classify_accuracy(np.array([1,0,1]), np.array([1, 1, 1])))
```
## 分类结果作图<a id="sec-2-2" name="sec-2-2"></a>

`注意：`
-   该画图方法是在内部训练进行画图，如果特征大于2，则降至2维再进行训练，而不是先训练后作图，因为要对图上每一个二维点都进行预测，因此，模型必须支持2维训练集（比如随机森林 m>2 时就不支持2维训练集）
-   如果想先训练再作图，且特征大于2维，则无法做出区域

范例：
```python
    from simple_ml import classify_plot
    classify_plot.classify_plot(model, X_train, y_train, X_test, Y_test, title='My Support Vector Machine')
```
## 交叉验证<a id="sec-2-3" name="sec-2-3"></a>

目前提供了两种交叉验证方法：

-   留出法（holdout）
-   k折法（k<sub>folder）</sub>

接受参数为：
1.  模型实例
2.  特征数据
3.  标签数据
4.  交叉验证类型
5.  训练样本比重：只针对留出法
6.  交叉验证次数

范例：
```python
    from simple_ml.cross_validation import *
    cross_validation(model, X, y, CrossValidationType.holdout, 0.3, 5)
```
# 分类算法<a id="sec-3" name="sec-3"></a>

## 类规范<a id="sec-3-1" name="sec-3-1"></a>

我在base.py 中给出了所有分类算法所虚继承的抽象类：BaseClassifier

主要作用是：
-   检查X，Y输入合法性
-   检查Y的类别，包括连续、二值、多值三种类型
-   申明样本数、变量数、训练集、测试集等类属性

必须要重写的方法有：
-   fit(X,Y) 给定数据集X和Y进行拟合
-   predict(X) 给定测试集进行预测
-   score(X,Y) 给定X，Y进行预测效果打分

## knn相关算法<a id="sec-3-2" name="sec-3-2"></a>

### 简单knn<a id="sec-3-2-1" name="sec-3-2-1"></a>

范例：
```python
    from simple_ml.knn import *
    from dataset.classify_data import get_iris
    knn_test = myKNN(K=3,distance_type=DisType.CosSim)
    X, y = get_iris()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    knn_test.fit(X_train, y_train)
    print(knn_test.predict(X_test))
    print(knn_test.score(X_test, y_test))
```
### KD树<a id="sec-3-2-2" name="sec-3-2-2"></a>

`Comming Soon`

## Logistic回归<a id="sec-3-3" name="sec-3-3"></a>

范例
```python
    from simple_ml.logistic import *
    X = np.array([[2,1], [4,2], [3,3], [4,1], [3,2], [2,3], [1,3]])
    y = np.array([1,2,0,1,0,1,2])
    lr = MyLogisticRegression(step=0.01,tol=1e-10)
    lr.fit(X, y)
    print(lr.predict(X))
    print(lr.score(X, y))
    lr.auc_plot(X, y)
```
## 贝叶斯相关算法<a id="sec-3-4" name="sec-3-4"></a>

### 朴素贝叶斯<a id="sec-3-4-1" name="sec-3-4-1"></a>

范例
```python
    from simple_ml.naive_bayes import *
    X = np.array([[0, 0, 0, 1],
               [0, 1, 0, 0],
               [1, 1, 0, 1],
               [0, 1, 1, 1],
               [0, 0, 0, 0]])
    y = np.array([0,1,0,1,0])
    nb = MyNaiveBayes()
    nb.fit(X, y)
    X_test = np.array([0, 0, 0, 0]).reshape(1, -1)
    print(nb.predict(X_test))
```
### 半朴素贝叶斯<a id="sec-3-4-2" name="sec-3-4-2"></a>

`Comming Soon`

### 贝叶斯最小误差<a id="sec-3-4-3" name="sec-3-4-3"></a>

注意：只支持离散标签
```python
import numpy as np
from simple_ml.bayes import MyBayesMinimumError

X = np.array([[2,1],
             [0,3],
             [3,0],
             [1,2],
             [2,0],
             [0,1.5]])
y = np.array([1,0,1,0,1,0])
bme = MyBayesMinimumError()
bme.fit(X, y)
print(bme.predict(X))
```


### 贝叶斯最小风险<a id="sec-3-4-4" name="sec-3-4-4"></a>
注意：只支持离散标签
```python
import numpy as np
from simple_ml.bayes import MyBayesMinimumRisk

X = np.array([[2,1],
             [0,3],
             [3,0],
             [1,2],
             [2,0],
             [0,1.5]])
y = np.array([1,0,1,0,1,0])
bme = MyBayesMinimumRisk(np.array([[0,10], [1,0]]))
bme.fit(X, y)
print(bme.predict(X))
```


## 基于树的算法<a id="sec-3-5" name="sec-3-5"></a>

### CART<a id="sec-3-5-1" name="sec-3-5-1"></a>

范例
```python
    from simple_ml.tree import *
    np.random.seed(1234)
    rt = RegressionTree(min_leaf_samples=3)
    X = np.random.rand(20, 10)
    Y = np.random.rand(20)
    y_test = np.random.rand(10)
    rt.fit(X, Y)
    print(rt.predict(y_test))
```
### 随机森林<a id="sec-3-5-2" name="sec-3-5-2"></a>

范例
```python
    from simple_ml.tree import *
    X, y = get_iris()
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    mrf = MyRandomForest(2)
    mrf.fit(X_train, y_train)
    print(mrf.predict(X_test))
    print(y_test)
    mrf.classifyPlot(X_test, y_test)
```
## 支持向量机<a id="sec-3-6" name="sec-3-6"></a>

-   暂时只支持二分类问题
-   提供核函数如下：
```python
    class KernelType(Enum):
        linear = 0      # 线性核
        polynomial = 1  # 多项式核
        gassian = 2     # 高斯核
        laplace = 3     # 拉普拉斯核
        sigmoid = 4     # sigmoid核
```
范例
```python
    from simple_ml.svm import *
    from simple_ml.classify_data import  get_iris
    X, y = get_iris()
    X = X[(y==1) | (y==2)]
    y = y[(y==1) | (y==2)]
    y = np.array([i if i ==1 else -1 for i in y])
    mysvm = MySVM(0.6, 0.001, 0.00001, 50, KernelType.linear)
    mysvm.fit(X, y)
    print(mysvm.alphas, mysvm.b)
    print(mysvm.predict(X))
    mysvm.classifyPlot(X, y)
```
## 神经网络<a id="sec-3-7" name="sec-3-7"></a>

### BP神经网络<a id="sec-3-7-1" name="sec-3-7-1"></a>

仅仅完成了单样本的情况

# 聚类<a id="sec-4" name="sec-4"></a>

## K均值聚类<a id="sec-4-1" name="sec-4-1"></a>

范例
```python
    from simple_ml.cluster import *
    X = np.array([1, 2,3, 5,6, 10,11,12,20, 35]).reshape(-1, 2)
    X = np.random.rand(*(50, 2))
    km = MyKMeans(3, DisType.Minkowski, d=2)
    km.fit(X)
    print(km.labels)
    # plot
    import matplotlib.pyplot as plt
    plt.scatter(x=X[:,0], y=X[:, 1], c=km.labels)
    plt.show()
```
## 层次聚类<a id="sec-4-2" name="sec-4-2"></a>

范例
```python
    from simple_ml.cluster import *
    X = np.array([1, 2,3, 5,6, 10,11,12,20, 35]).reshape(-1, 2)
    X = np.random.rand(*(50, 2))
    km = MyHierarchical(DisType.Minkowski, d=2)
    km.fit(X)
    print(km.max_dis)
    print(km.cluster(km.max_dis/4))
    # plot
    import matplotlib.pyplot as plt
    plt.scatter(x=X[:,0], y=X[:, 1], c=km.labels)
    plt.show()
```

# Boosting学习<a id="sec-5" name="sec-5"></a>
## AdaBoost<a id="sec-5-1" name="sec-5-1"></a>
```python
from simple_ml.ensemble import MyAdaBoost
import numpy as np
X = np.array([[2,1], [4,2], [3,3], [4,1], [3,2], [2,3], [1,3]])
y = np.array([1,2,0,1,0,1,2])
lr = MyAdaBoost(nums=10)
lr.fit(X, y)
lr.predict(X)
```
## GBDT<a id="sec-5-2" name="sec-5-2"></a>
- 只支持0-1特征
- 只支持连续标签
- 只支持平方损失

```python
from simple_ml.ensemble import *

X = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1]).reshape(2, -1).T
y = np.array([3., 3.2, 2., 2.1, 1.5, 2.3, 1.4, 2.1])
gbdt = MyGBDT()
gbdt.fit(X, y)
print(gbdt.predict(np.array([[1, 1], [0, 0], [1, 0], [0, 1]])))
```



`Losers Always Whine About Their Best`

`献给所有为梦想不懈奋斗的人儿们`

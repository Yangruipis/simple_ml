#+LATEX_HEADER: \newenvironment{lequation}{\begin{equation}\Large}{\end{equation}}
#+ATTR_LATEX: :width 5cm :options angle=90
#+TITLE: 一个简单的机器学习实现
#+AUTHOR: 杨 睿
#+EMAIL: yangruipis@163.com
#+KEYWORDS: Machine Learning
#+OPTIONS: H:4 toc:t 
#+ATTR_HTML: title="Codacy Badge"

将机器学习的基本流程与算法进行手写实现，仅调用numpy以及python基本库

[[https://img.shields.io/npm/l/express.svg]] 

[[https://www.codacy.com/app/Yangruipis/simpleML?utm_source=github.com&utm_medium=referral&utm_content=Yangruipis/simpleML&utm_campaign=badger][file:https://api.codacy.com/project/badge/Grade/6c44426cd41f4b0382ca5714d97f56fe]]

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/00c639db60364d12b0102456552fe806)](https://www.codacy.com/app/Yangruipis/simpleML?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Yangruipis/simpleML&amp;utm_campaign=Badge_Grade)

* 特征工程


  
** 特征预处理
包括缺失值、异常值、标准化等等，由于比较简单，故跳过
** 特征选择
*** Filter方法

当前提供了四种Filter选择方法：

- 方差法
- 相关系数法
- 卡方检验法
- 互信息法

范例如下
#+BEGIN_SRC python
from simple_ml.filter_select import *
X = np.random.random(20).reshape(-1, 4)
Y = np.random.randint(0,2,5)
mf = MyFilter(filter_type=FilterType.chi2, top_k=3)
mf.fitTransform(X,Y)
mf.transform(X)
#+END_SRC
  
* 模型评价
** 相关得分
*** 针对二分类问题：
    - accuracy
    - precision
    - recall
    - f1
    - auc
    - roc作图
*** 针对多分类问题：
    - f1_micro
    - f1_macro
    - f1_weight
*** 针对回归问题：
    - explained_variance
    - absolute_error
    - squared_error
    - RMSE(root mean squared error)
    - RMSLE(root mean squared log error, in case of the abnormal value)
    - r2
    - median_absolute_error

范例：
#+BEGIN_SRC python
from simple_ml.score import *
print(classify_accuracy(np.array([1,0,1]), np.array([1, 1, 1])))
#+END_SRC

** 分类结果作图

~注意：~
- 该画图方法是在内部训练进行画图，如果特征大于2，则降至2维再进行训练，而不是先训练后作图，因为要对图上每一个二维点都进行预测，因此，模型必须支持2维训练集（比如随机森林 m>2 时就不支持2维训练集）
- 如果想先训练再作图，且特征大于2维，则无法做出区域

范例：
#+BEGIN_SRC python
from simple_ml import classify_plot
classify_plot.classify_plot(model, X_train, y_train, X_test, Y_test, title='My Support Vector Machine')
#+END_SRC

** 交叉验证

目前提供了两种交叉验证方法：

- 留出法（holdout）
- k折法（k_folder）

接受参数为：
1. 模型实例
2. 特征数据
3. 标签数据
4. 交叉验证类型
5. 训练样本比重：只针对留出法
6. 交叉验证次数

范例：
#+BEGIN_SRC python
from simple_ml.cross_validation import *
cross_validation(model, X, y, CrossValidationType.holdout, 0.3, 5)
#+END_SRC 

* 分类算法
** 类规范
我在abstract.myclassifier.py 中给出了所有分类算法所虚继承的抽象类：myClassifier

主要作用是：
- 检查X，Y输入合法性
- 检查Y的类别，包括连续、二值、多值三种类型
- 申明样本数、变量数、训练集、测试集等类属性

必须要重写的方法有：
- fit(X,Y) 给定数据集X和Y进行拟合
- predict(X) 给定测试集进行预测
- score(X,Y) 给定X，Y进行预测效果打分

** knn相关算法
*** 简单knn
范例：

#+BEGIN_SRC python
  from simple_ml.knn import *
  from dataset.classify_data import get_iris
  knn_test = myKNN(K=3,distance_type=DisType.CosSim)
  X, y = get_iris()
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
  knn_test.fit(X_train, y_train)
  print(knn_test.predict(X_test))
  print(knn_test.score(X_test, y_test))
#+END_SRC

*** KD树
~Comming Soon~

** Logistic回归

范例

#+BEGIN_SRC python
 from simple_ml.logistic import *
 X = np.array([[2,1], [4,2], [3,3], [4,1], [3,2], [2,3], [1,3]])
 y = np.array([1,2,0,1,0,1,2])
 lr = MyLogisticRegression(step=0.01,tol=1e-10)
 lr.fit(X, y)
 print(lr.predict(X))
 print(lr.score(X, y))
 lr.auc_plot(X, y)
#+END_SRC

** 贝叶斯相关算法

*** 朴素贝叶斯
范例

#+BEGIN_SRC python
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
#+END_SRC

*** 半朴素贝叶斯
~Comming Soon~
** 基于树的算法

*** CART 

范例
#+BEGIN_SRC python
from simple_ml.tree import *
np.random.seed(1234)
rt = RegressionTree(min_leaf_samples=3)
X = np.random.rand(20, 10)
Y = np.random.rand(20)
y_test = np.random.rand(10)
rt.fit(X, Y)
print(rt.predict(y_test))
#+END_SRC

*** 随机森林

范例

#+BEGIN_SRC python
from simple_ml.tree import *
X, y = get_iris()
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
mrf = MyRandomForest(2)
mrf.fit(X_train, y_train)
print(mrf.predict(X_test))
print(y_test)
mrf.classifyPlot(X_test, y_test)
#+END_SRC

** 支持向量机

- 暂时只支持二分类问题
- 提供核函数如下：
#+BEGIN_SRC python
class KernelType(Enum):
    linear = 0      # 线性核
    polynomial = 1  # 多项式核
    gassian = 2     # 高斯核
    laplace = 3     # 拉普拉斯核
    sigmoid = 4     # sigmoid核
#+END_SRC

范例

#+BEGIN_SRC python
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
#+END_SRC

** 神经网络
*** BP神经网络
仅仅完成了单样本的情况
* 聚类
** K均值聚类
范例

#+BEGIN_SRC python
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
#+END_SRC


** 层次聚类
范例

#+BEGIN_SRC python
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
#+END_SRC

=Losers Always Whine About Their Best=

~献给所有为梦想不懈奋斗的人儿们~





Simple Meachine Learning
一个简单的机器学习算法实现

![](https://img.shields.io/npm/l/express.svg)  [![Codacy Badge](https://api.codacy.com/project/badge/Grade/00c639db60364d12b0102456552fe806)](https://www.codacy.com/app/Yangruipis/simpleML?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Yangruipis/simpleML&amp;utm_campaign=Badge_Grade) [![Join the chat at https://gitter.im/simple_ml/Lobby](https://badges.gitter.im/simple_ml/Lobby.svg)](https://gitter.im/simple_ml/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

---


# 快速开始

## 安装

**环境和依赖库**
- python3.5及以上
- windows or Linux
- numpy
- matplotlib
`强烈推荐Anaconda环境`

**终端安装**

```bash
git clone https://github.com/Yangruipis/simple_ml.git
cd ./simple_ml
pip install setup.py
```

## 使用

```python
# 一个简单的例子，用CART树进行二分类
from simple_ml.tree import CART
import numpy as np
X = np.array([[1,1.1],
              [1,2.0],
              [0,3.0],
              [0,2.2]])
y = np.array([1,1,0,0])
cart = CART(min_samples_leaf=1)
cart.fit(X, y)
x_test = np.array([[1,2],[3,4]])
print(cart.predict(x_test))
```
```python
>>> np.array([1,1])
```

`./simple_ml/examples`文件夹中提供了大多数方法的使用范例，更详细的用法见 [帮助手册 manual.md](./manual.md)

# 它能做什么

## 最最最最主要的任务

如果你同时满足：
1. **机器学习入门阶段**
2. **python 进阶阶段**

那么恭喜你，这个项目可以给你提供如下帮助：

- **阅读源码**， 不像sklearn过于复杂难读的源码，这个轻量级的项目非常易读，并且我尽可能的增加了注释，提高代码的可读性
- **学习知识**，该项目梳理基本机器学习算法的种类和流程，工程实现上的大致步骤，中间出现的一些细节问题以及如何解决
- **实时交流**，我在 gitter 上建立了 [gitchat 聊天室](https://gitter.im/simple_ml/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)，欢迎大家就项目本身的具体问题，或者其他任何相关事项进行讨论，欢迎大家积极提 issues，我会第一时间回复


## 作为一个机器学习项目的任务

### 1. 特征

### 2. 分类和回归

#### 2.1 二分类
`simple_ml`提供了非常多的二分类方法，以[wine数据集](http://archive.ics.uci.edu/ml/datasets/Wine)为例（见`./simple_ml/examples`），分类效果和方法名称见下图。
![pic1](./doc/imgs/wine.jpg)

#### 2.2 多分类

`simple_ml`暂时只提供了一些多分类算法，见下图，同样是[wine数据集](http://archive.ics.uci.edu/ml/datasets/Wine)，后面作者将会进行补充。

![pic2](./doc/imgs/wine2.jpg)

#### 2.3 回归

`simple_ml`提供了`CART`、`GBDT`这两种回归方法，后面将加入`SVR`


### 3. 聚类


`注:`以上所有图均为simple_ml直出（需要matplotlib）


# 为什么会有这个项目 & 致谢

作者本科在上海一个双非商科院校读统计，而后保研失败继续在本校读经济，从大二开始接触机器学习，以及编程相关知识（我的轨迹：stata->R->C#->python)，对数据和编程非常感兴趣，基本上一路走过来全靠自学。

而现在找工作的路磕磕绊绊（个人能力+非科班非211非985），有可能以后也不会从事算法工程师相关工作，但是总想留下一点东西，尤其是即将毕业之际。以后看起来可能非常可笑吧，不过总归是曾经的轨迹。

作者在接下来的一年找工作的同时，将尽全力维护该项目，不断更新和修改，热烈欢迎任何贡献和讨论。

**致谢：**
- 首先感谢我自己，一路走来的不易如人饮水
- 其次感谢我的好友[何燕杰](https://github.com/YanjieHe)和[程刚](https://github.com/chenggang0815)对我在学习和工作上的帮助
- 最后感谢所有相关书籍、博客的作者


# TODO list:

- [ ] test cases
- [ ] an efficient bp network 
- [ ] more optimal methods
- [x] train test split func in helper
- [x] other feature select method to add
- [x] lasso and Ridge
- [x] add GBDT feature select
- [x] update Readme
- [ ] setup.py
- [x] examples
- [x] get more datasets
- [ ] regression plot
- [ ] kd_tree
- [ ] Support Machine Regression
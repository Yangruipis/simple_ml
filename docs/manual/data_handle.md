# 数据处理模块 **simple_ml.data_handle**

- [读取字符串](#%E8%AF%BB%E5%8F%96%E5%AD%97%E7%AC%A6%E4%B8%B2)
- [读取csv](#%E8%AF%BB%E5%8F%96csv)
- [数据编码(Encoding)](#%E6%95%B0%E6%8D%AE%E7%BC%96%E7%A0%81encoding)
- [获取变量类型](#%E8%8E%B7%E5%8F%96%E5%8F%98%E9%87%8F%E7%B1%BB%E5%9E%8B)
- [异常值处理](#%E5%BC%82%E5%B8%B8%E5%80%BC%E5%A4%84%E7%90%86)
- [缺失值处理](#%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86)
- [独热编码(One-hot Encoding)](#%E7%8B%AC%E7%83%AD%E7%BC%96%E7%A0%81one-hot-encoding)
- [全自动处理](#%E5%85%A8%E8%87%AA%E5%8A%A8%E5%A4%84%E7%90%86)
- [随机数据集切分](#%E9%9A%8F%E6%9C%BA%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%87%E5%88%86)
- [返回](#%E8%BF%94%E5%9B%9E)


* * *

## 读取字符串

```python
def read_string(string, header=True, index=True, sep=",")
```

读取字符串为二维数组，如果某列是整型，转为int，如果是小数，转为float，如果是文本，则不变


|             |  名称  |    类型     |      描述       |
|------------:|:------:|:-----------:|:---------------:|
| Parameters: | string |     str     | 需要读取的字符串 |
|             | header |    bool     | 第一行是否为标题 |
|             | index  |    bool     | 第一列是否为索引 |
|             |  sep   |     str     |      分隔符      |
|    Returns: |        | np.2dnarray |     二维数组     |

## 读取csv

```python
def read_csv(path, header=True, index=True, sep=",")
```

读取csv文件

|             |  名称  |     类型     |      描述       |
|------------:|:------:|:------------:|:---------------:|
| Parameters: |  path  |     str      |     文件路径     |
|             | header |     bool     | 第一行是否为标题 |
|             | index  |     bool     | 第一列是否为索引 |
|             |  sep   |     str      |      分隔符      |
|    Returns: |        | List[List[]] |     二维列表     |


## 数据编码(Encoding)


```python
def number_encoder(x_lst)
```

将含有文本、整型、浮点型的数据进行编码，统一格式为浮点型

|             | 名称  |     类型     |           描述           |
|------------:|:-----:|:------------:|:------------------------:|
| Parameters: | x_lst | List[List[]] | 二维列表数据，任意类型均可 |
|    Returns: |       |  np.2darray  |       二维numpy数组       |

## 获取变量类型

```python
def get_type(arr)
```

获取一个二维数组每一列的变量类型，包括了两类(binary)、多类(multi_class)、连续(continuous)三种类型

|             | 名称 |       类型       |     描述      |
|------------:|:----:|:---------------:|:------------:|
| Parameters: | arr  |   np.2darray    | 二维numpy数组 |
|    Returns: |      | List[[LabelType](../structure/enum.md)] | 标签类型的列表 |


## 异常值处理

```python
def abnormal_handle(arr, type_list, up=90, lp=10)
```

通过winsorize变换进行异常值处理，即将超过上或者下分位数的样本用该分位数进行替换

|             |   名称    |      类型       |           描述            |
|------------:|:---------:|:---------------:|:-------------------------:|
| Parameters: |    arr    |   np.2darray    | 需要进行异常值处理的二维数组 |
|             | type_list | List[[LabelType](../structure/enum.md)] |       标签类型的列表       |
|             |    up     | int, (50, 100)  |          上分位数          |
|             |    lp     |    int(0,50)    |          下分位数          |
|    Returns: |           |   np.2darray    |      处理好的二维列表      |


## 缺失值处理

```python
from simple_ml.base.base_enum import ConMissingHandle, DisMissingHandle

def missing_value_handle(arr, type_list, continuous_method=ConMissingHandle.mean_fill,
                         discrete_method=DisMissingHandle.mode_fill)
```

- 对离散变量和连续变量采取不同的方法进行 **缺失值处理**
- 必须先 **异常值处理**，再**缺失值处理**，否则填补的变量将会包含异常值信息

|             |       名称        |       类型       |           描述            |
|------------:|:-----------------:|:----------------:|:-------------------------:|
| Parameters: |        arr        |    np.2darray    | 需要进行缺失值处理的二维数组 |
|             |     type_list     | List[[LabelType](../structure/enum.md)]  |       标签类型的列表       |
|             | continuous_method |  [ConMissingHandle](../structure/enum.md) |     连续数据的处理方法      |
|             |  discrete_method  |  [DisMissingHandle](../structure/enum.md) |     离散数据的处理方法      |
|    Returns: |                   |    np.2darray    |      处理好的二维列表      |


## 独热编码(One-hot Encoding)

```python
def one_hot_encoder(arr, type_list)
```

针对多分类变量进行独热编码，转为虚拟变量，**虚拟变量陷阱** 已经进行了处理


|        名称 |   类型    |      描述       |                           |
|------------:|:---------:|:---------------:|:-------------------------:|
| Parameters: |    arr    |   np.2darray    | 需要进行缺失值处理的二维数组 |
|             | type_list | List[[LabelType](../structure/enum.md)] |       标签类型的列表       |
|    Returns: |           |   np.2darray    |      处理好的二维列表      |


## 全自动处理

```python
def BIGMOM(path, header=True, index=True, sep=",")
```

只需提供csv文件位置，即可一键处理

|             |  名称  |    类型    |       描述       |
|------------:|:------:|:----------:|:----------------:|
| Parameters: |  path  |    str     |     文件位置      |
|             | header |    bool    | 第一行是否为变量名 |
|             | index  |    bool    | 第一列是否为索引  |
|             |  sep   |    str     |      切割符       |
|    Returns: |        | np.2darray | 处理好的两维数组  |

## 随机数据集切分

```python
def train_test_split(x, y, test_size=0.3, seed=None)
```

对给定特征和标签进行随机切分，返回切分后的训练集和测试集

|             |   名称    |                类型                |           描述            |
|------------:|:---------:|:----------------------------------:|:-------------------------:|
| Parameters: |     x     |             np.2darray             |           特征集           |
|             |     y     |              np.array              |           标签集           |
|             | test_size |            float, (0,1)            |       测试集样本占比       |
|             |   seed    |                int                 | 随机种子值（保证试验可重复） |
|    Returns: |           | (x_train, y_train, x_test, y_test) |   分割后的测试集和训练集    |


## 获取k折后的配对样本下标

```python
def get_k_folder_idx(length, k_folder, seed=918):
    """
    获取k折后的配对样本下标
    :param length:    样本长度
    :param k_folder:  K折数目
    :param seed:      随机种子
    :return:          迭代器， (其中一个folder下标，剩余folder下标)
    """
    pass
```

|             |   名称   |      类型       |                描述                |
|------------:|:--------:|:---------------:|:----------------------------------:|
| Parameters: |  length  |       int       |              样本长度               |
|             | k_folder |       int       |               k折数目               |
|             |   seed   |       int       |              随机种子               |
|    Returns: |          | iterator(tuple) | (其中一个folder下标，剩余folder下标) |


# [返回主页](../index.md)

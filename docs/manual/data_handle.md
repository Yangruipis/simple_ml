
{% capture markdown %}

# 数据处理模块 **simple_ml.data_handle**


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
|    Returns: |      | List[LabelType] | 标签类型的列表 |


## 异常值处理

```python
def abnormal_handle(arr, type_list, up=90, lp=10)
```

通过winsorize变换进行异常值处理，即将超过上或者下分位数的样本用该分位数进行替换

|             |   名称    |      类型       |           描述            |
|------------:|:---------:|:---------------:|:-------------------------:|
| Parameters: |    arr    |   np.2darray    | 需要进行异常值处理的二维数组 |
|             | type_list | List[LabelType] |       标签类型的列表       |
|             |    up     | int, (50, 100)  |          上分位数          |
|             |    lp     |    int(0,50)    |          下分位数          |
|    Returns: |           |   np.2darray    |      处理好的二维列表      |


## 缺失值处理

```python
from simple_ml.ensemble import ConMissingHandle, DisMissingHandle
def missing_value_handle(arr, type_list, continuous_method=ConMissingHandle.mean_fill,
                         discrete_method=DisMissingHandle.mode_fill)
```

- 对离散变量和连续变量采取不同的方法进行 **缺失值处理**
- 必须先 **异常值处理**，再**缺失值处理**，否则填补的变量将会包含异常值信息

|             |       名称        |       类型       |           描述            |
|------------:|:-----------------:|:----------------:|:-------------------------:|
| Parameters: |        arr        |    np.2darray    | 需要进行缺失值处理的二维数组 |
|             |     type_list     | List[LabelType]  |       标签类型的列表       |
|             | continuous_method | ConMissingHandle |     连续数据的处理方法      |
|             |  discrete_method  | DisMissingHandle |     离散数据的处理方法      |
|    Returns: |                   |    np.2darray    |      处理好的二维列表      |


## 独热编码(One-hot Encoding)

```python
def one_hot_encoder(arr, type_list)
```

针对多分类变量进行独热编码，转为虚拟变量，**虚拟变量陷阱** 已经进行了处理


|        名称 |   类型    |      描述       |                           |
|------------:|:---------:|:---------------:|:-------------------------:|
| Parameters: |    arr    |   np.2darray    | 需要进行缺失值处理的二维数组 |
|             | type_list | List[LabelType] |       标签类型的列表       |
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


# [返回](../index.md)

{: #custom-heading} {% endcapture %} {% assign text = markdown | markdownify %}

{% include toc.html html=text %}

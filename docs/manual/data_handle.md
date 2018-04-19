
**数据处理模块**

```python
from simple_ml.data_handle import *
```


* * *

# 读取字符串

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

# 读取csv

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


# 数据编码(Encoding)


```python
def number_encoder(x_lst)
```

将含有文本、整型、浮点型的数据进行编码，统一格式为浮点型

|             | 名称  |     类型     |           描述           |
|------------:|:-----:|:------------:|:------------------------:|
| Parameters: | x_lst | List[List[]] | 二维列表数据，任意类型均可 |
|    Returns: |       |  np.2darray  |       二维numpy数组       |

# 获取变量类型

```python
def get_type(arr)
```

获取一个二维数组每一列的变量类型，包括了两类(binary)、多类(multi_class)、连续(continuous)三种类型

|             | 名称 |       类型       |      描述       |
|------------:|:----:|:---------------:|:--------------:|
| Parameters: | arr  |   np.2darray    |  二维numpy数组  |
|    Returns: |      | List[LabelType] | 标签枚举类的列表 |

# 独热编码(One-hot Encoding)

# 缺失值处理

# 异常值处理


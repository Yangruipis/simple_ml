

# 结构体 simple_ml.base.base_model

- [结构体 simple_ml.base.base_model](#%E7%BB%93%E6%9E%84%E4%BD%93-simplemlbasebasemodel)
    - [二叉树节点](#%E4%BA%8C%E5%8F%89%E6%A0%91%E8%8A%82%E7%82%B9)
    - [多叉树节](#%E5%A4%9A%E5%8F%89%E6%A0%91%E8%8A%82)
    - [GBDT二叉树节点](#gbdt%E4%BA%8C%E5%8F%89%E6%A0%91%E8%8A%82%E7%82%B9)
- [返回主页](#%E8%BF%94%E5%9B%9E%E4%B8%BB%E9%A1%B5)

* * *

## 二叉树节点

```python
class BinaryTreeNode:

    def __init__(self, left=None, right=None, data_id=None, 
    feature_id=None, value=None, leaf_label=None):
        if data_id is None:
            data_id = []
        self.left = left
        self.right = right
        self.data_id = data_id
        self.feature_id = feature_id
        self.value = value
        self.leaf_label = leaf_label
```

|             |    名称    |       类型       |              描述              |
|------------:|:----------:|:----------------:|:------------------------------:|
| Parameters: |    left    |  BinaryTreeNode  |             左节点              |
|             |   right    |  BinaryTreeNode  |             右节点              |
|             |  data_id   | List or np.array |           样本id列表            |
|             | feature_id |       int        |             特征id             |
|             |   value    |   float or int   |          对应特征的值           |
|             | leaf_label |   float or int   | 叶子节点的标签（非叶子节点为None) |
|    Returns: |            |       Void       |                                |


## 多叉树节

```python
class MultiTreeNode:

    def __init__(self, child=None, data_id=None, feature_id=None, value=None, leaf_label=None):
        if data_id is None:
            data_id = []
        self.child = child
        self.data_id = data_id
        self.feature_id = feature_id
        self.value = value
        self.leaf_label = leaf_label
```


|             |    名称    |        类型         |              描述              |
|------------:|:----------:|:-------------------:|:------------------------------:|
| Parameters: |   child    | List[MultiTreeNode] |          孩子节点列表           |
|             |  data_id   |  List or np.array   |           样本id列表            |
|             | feature_id |         int         |             特征id             |
|             |   value    |    float or int     |          对应特征的值           |
|             | leaf_label |    float or int     | 叶子节点的标签（非叶子节点为None) |
|    Returns: |            |        Void         |                                |


## GBDT二叉树节点

```python
from simple_ml.base.base_model import BinaryTreeNode

class GBDTTreeNode(BinaryTreeNode):

    def __init__(self, left=None, right=None, data_id=None, feature_id=None, value=None, leaf_label=None):
        super(GBDTTreeNode, self).__init__(left, right, data_id, feature_id, value, leaf_label)
        self.gamma = None
```

> 继承`BinaryTreeNode`
>
> gamma值用以记录每个叶子节点的最优值

|             |    名称    |       类型       |              描述              |
|------------:|:----------:|:----------------:|:------------------------------:|
| Parameters: |    left    |  BinaryTreeNode  |             左节点              |
|             |   right    |  BinaryTreeNode  |             右节点              |
|             |  data_id   | List or np.array |           样本id列表            |
|             | feature_id |       int        |             特征id             |
|             |   value    |   float or int   |          对应特征的值           |
|             | leaf_label |   float or int   | 叶子节点的标签（非叶子节点为None) |
|             |   gamma    |   float or int   |        叶子节点的最优值         |
|    Returns: |            |       Void       |                                |

# [返回主页](../../index.md)

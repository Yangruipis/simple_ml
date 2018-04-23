# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

from simple_ml.base.base_error import EmptyInputError, MissingHandleTypeError, FeatureNumberMismatchError
from simple_ml.base.base_enum import DisMissingHandle, ConMissingHandle, LabelType
import numpy as np
from collections import Counter


__all__ = [
    'read_string',
    'read_csv',
    'number_encoder',
    'get_type',
    'abnormal_handle',
    'missing_value_handle',
    'one_hot_encoder',
    'BIGMOM',
    'train_test_split',
    'transform_y',
    'DisMissingHandle',
    'ConMissingHandle',
    'get_k_folder_idx',
]


def _is_number(s):
    """
    判断一个字符串是否是数字
    :param s:  字符串
    :return:   bool
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def read_string(string, header=True, index=True, sep=","):
    """
    读取字符串为二维数组，如果某列是整型，转为int，如果是小数，转为float，如果是文本，则不变
    :param string:  字符串
    :param header:  第一行是否为列名
    :param index:   第一列是否为索引
    :param sep:     分隔符
    :return:        处理好的二维数组
    """
    if string.strip() == "":
        raise EmptyInputError("输入文本为空")

    res = [i.split(sep) for i in string.strip().replace(" ", "").split("\n")]
    if header:
        res = res[1:]
    if index:
        res = [i[1:] for i in res]

    for i in range(len(res[0])):
        is_float = True
        is_int = True
        for j, _res in enumerate(res):
            if not str.isdigit(res[j][i]):
                is_int = False
            if not _is_number(res[j][i]):
                is_float = False

        if is_int:
            for j, _res in enumerate(res):
                if res[j][i] == "" or res[j][i] == "?" or res[j][i].lower() == "nan":
                    res[j][i] = np.nan
                else:
                    res[j][i] = int(res[j][i])
        elif is_float:
            for j, _res in enumerate(res):
                if res[j][i] == "" or res[j][i] == "?" or res[j][i].lower() == "nan":
                    res[j][i] = np.nan
                else:
                    res[j][i] = float(res[j][i])

    return res


def read_csv(path, header=True, index=True, sep=","):
    with open(path, 'r') as f:
        string = f.read()
    return read_string(string.strip(), header, index, sep)


def number_encoder(x_lst):
    """
    数据编码，将中文转为代表类别的数字
    :param x_lst: read_string处理后的二维列表
    :return: np.ndarray
    """
    if len(x_lst) == 0 or len(x_lst[0]) == 0:
        raise EmptyInputError("输入数组为空")

    l1, l2 = len(x_lst), len(x_lst[0])
    res = np.zeros((l1, l2))
    for j in range(l2):
        column = []
        for i in range(l1):
            column.append(x_lst[i][j])
        if isinstance(x_lst[0][j], str):
            count = Counter(column)
            map_list = {s: i for i, s in enumerate(count.keys())}
            res[:, j] = list(map(lambda x: map_list[x], column))
        else:
            res[:, j] = column
    return res


def get_type(arr):
    """
    获取变量类型数组，包括了binary， multi_class，continuous
    :param arr:   np.ndarray
    :return:      list[LabelType]
    """
    res = []
    for feature in arr.T:
        count = np.unique([i for i in feature if not np.isnan(i)])
        is_continuous = False
        for i in count:
            # 当存在浮点型，且小数点后有数字时，是连续值
            if int(i) != i:
                is_continuous = True

        if len(count) == 2:
            res.append(LabelType.binary)
        elif is_continuous:
            res.append(LabelType.continuous)
        else:
            res.append(LabelType.multi_class)
    return res


def abnormal_handle(arr, type_list, up=90, lp=10):
    """
    在缺失值处理之前做，否则补时全会用到异常值信息
    :param arr:        ndarray
    :param type_list:  list[LabelType]
    :return:           ndarray
    """
    if arr.shape[1] != len(type_list):
        raise FeatureNumberMismatchError
    arr = arr.copy()
    for i, _type in enumerate(type_list):
        if _type == LabelType.continuous:
            arr[:, i] = _winsorize(arr[:, i], up, lp)
    return arr


def _winsorize(arr, upper_percentage=95, lower_percentage=5):
    upper = np.nanpercentile(arr, upper_percentage)
    lower = np.nanpercentile(arr, lower_percentage)
    res = []
    for i in arr:
        if not np.isnan(i):
            if i > upper:
                res.append(upper)
            elif i < lower:
                res.append(lower)
            else:
                res.append(i)
        else:
            res.append(np.nan)
    return np.array(res)


def missing_value_handle(arr, type_list, continuous_method=ConMissingHandle.mean_fill,
                         discrete_method=DisMissingHandle.mode_fill):
    """
    缺失值处理
    必须先经过缺失值处理再去进行OneHotEncode
    :param arr:                 ndarray
    :param type_list:           list[LabelType]
    :param continuous_method:   连续数据的缺失值处理方法
    :param discrete_method:     离散数据的缺失值处理方法
    :return:                    ndarray
    """
    if arr.shape[1] != len(type_list):
        raise FeatureNumberMismatchError
    drop_sample_idx = []
    arr = arr.copy()
    for i, _type in enumerate(type_list):

        if _type == LabelType.continuous:
            if continuous_method == ConMissingHandle.mean_fill:
                mean = np.nanmean(arr[:, i])
                arr[:, i] = [mean if np.isnan(num) else num for num in arr[:, i]]
            elif continuous_method == ConMissingHandle.median_fill:
                median = np.nanmedian(arr[:, i])
                arr[:, i] = [median if np.isnan(num) else num for num in arr[:, i]]
            elif continuous_method == ConMissingHandle.sample_drop:
                drop_sample_idx += [j for j, num in enumerate(arr[:, i]) if np.isnan(num)]
            else:
                raise MissingHandleTypeError
        else:
            if discrete_method == discrete_method.mode_fill:
                count_dic = dict(Counter(arr[:, i]))
                mode = max(count_dic.keys(), key=lambda x: count_dic[x])
                arr[:, i] = [mode if np.isnan(num) else num for num in arr[:, i]]
            elif discrete_method == discrete_method.one_hot:
                other_code = np.max(arr[:, i]) + 1
                arr[:, i] = [other_code if np.isnan(num) else num for num in arr[:, i]]
            elif discrete_method == discrete_method.sample_drop:
                drop_sample_idx += [j for j, num in enumerate(arr[:, i]) if np.isnan(num)]
            else:
                raise MissingHandleTypeError
    if not drop_sample_idx:
        return arr
    else:
        return arr[list(set(list(range(arr.shape[0]))) - set(drop_sample_idx))]


def one_hot_encoder(arr, type_list):
    """
    独热编码，尤其针对多分类变量
    注意：需要提前进行缺失值处理，确保内部没有缺失值
    :param arr:           np.ndarray
    :param type_list:     list[LabelType]
    :return:              np.ndarray
    """
    if arr.shape[1] != len(type_list):
        raise FeatureNumberMismatchError
    res = []
    for i, _type in enumerate(type_list):
        if _type == LabelType.binary:
            res.append(list(map(int, arr[:, i] == arr[0, i])))
        elif _type == LabelType.multi_class:
            unique_value = np.unique(arr[:, i])
            for j in unique_value[1:]:
                res.append(list(map(int, arr[:, i] == j)))
        else:
            res.append(list(arr[:, i]))
    return np.array(res).T


def BIGMOM(path, header=True, index=True, sep=","):
    """

    老母亲函数，帮你操办一切，啥都不愁

    example：
        >>> path = "./test/adult.txt"
        >>> res= BIGMOM(path, False, True)

    :param path:     文本文件路径
    :return:         np.2darray
    """
    lst = read_csv(path, header, index, sep)
    arr = number_encoder(lst)
    types = get_type(arr)
    arr = abnormal_handle(arr, types)
    arr = missing_value_handle(arr, types)
    arr = one_hot_encoder(arr, types)
    return arr


def train_test_split(x, y, test_size=0.3, seed=None):
    """
    数据集随机切分
    :param x:  特征 feature, np.2darray
    :param y:  标签 label,   np.array
    :param test_size: 测试集样本数占比
    :param seed:      随机种子
    :return:          x_train, y_train, x_test, y_test
    """
    if seed:
        np.random.seed(seed)
    id_list = np.arange(x.shape[0])
    id_train = np.random.choice(id_list, int(len(id_list)*(1-test_size)), replace=False)
    id_test = np.array([i for i in id_list if i not in id_train])
    return x[id_train, :], y[id_train], x[id_test, :], y[id_test]


def transform_y(y):
    """
    转换y
    """
    if list(np.unique(y)) == [-1, 1]:
        return np.array([0 if i == -1 else i for i in y])
    else:
        return y


def get_k_folder_idx(length, k_folder, seed=918):
    """
    获取k折后的配对样本下标
    :param length:    样本长度
    :param k_folder:  K折数目
    :param seed:      随机种子
    :return:          迭代器， (其中一个folder下标，剩余folder下标)
    """
    arr = np.arange(length)
    np.random.seed(seed)
    random_arr = np.random.choice(arr, length, False)
    group_list = np.array([i % k_folder for i in arr])
    for i in range(k_folder):
        yield (random_arr[group_list == i], random_arr[group_list != i])

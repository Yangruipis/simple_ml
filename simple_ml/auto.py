# -*- coding:utf-8 -*-

"""
The Kernel Module of simple_ml
"""

from simple_ml.base import *
from simple_ml.data_handle import *
from simple_ml.helper import *
import numpy as np
import os


def _get_grid(dic: dict):
    res_id = []
    each_param_len = [len(i) for i in dic.values()]
    _get_pairs(each_param_len, 0, res_id)
    for ids in res_id:
        yield {i: dic[i][ids[j]] for j, i in enumerate(dic)}


def _get_pairs(arr, i, res):
    if i >= len(arr):
        res.append(arr[:])
    else:
        count = arr[i]
        for val in range(count):
            arr[i]= val
            _get_pairs(arr, i+1, res)
        arr[i] = count


def grid_search(func, params: dict, cv_times=5, increase=True, quiet=True):
    if increase:
        best = (None, -np.inf)
    else:
        best = (None, np.inf)

    for param in _get_grid(params):
        score_list = []
        for i in range(cv_times):
            score_list.append(func(**param))
        score = np.mean(score_list)
        if not quiet:
            print("param: %s, score: %a" % (param, score))
        if increase:
            if score > best[1]:
                best = (param, score)
        else:
            if score < best[1]:
                best = (param, score)
    print("BEST: param: %s, score: %a" % best)
    return best


def random_search():
    """
    随机搜索，包括了模拟退火，粒子群，等等， 后面慢慢写，由于simple_ml提供的参数较少，没有必要用到随机搜索
    :return:
    """
    pass

def bayes_search():
    """
    贝叶斯搜索，通过高斯过程实现
    :return:
    """
    pass


class BaseAuto:

    pass

class AutoDataHandle(BaseAuto):

    def __init__(self, cv_times=1):
        self.cv_times = cv_times
        self._data = None
        self._types = None
        self._data_handled = None
        self._head = None

    @property
    def origin_data(self):
        return self._data

    @property
    def handled_data(self):
        return self._data_handled

    @property
    def type_list(self):
        return self._types

    def read_from_file(self, path, header, index, sep):
        if os.path.exists(path):
            if header:
                self._head = get_head_list(path, sep)
            txt_list = read_csv(path, header, index, sep)
            self._data = number_encoder(txt_list)
            self._types = get_type(self._data)
        else:
            raise FileExistsError("%s 文件不存在" % path)

    def read_from_str(self, string, header, index, sep):
        if isinstance(string, str):
            txt_list = read_string(string, header, index, sep)
            self._data = number_encoder(txt_list)
            self._types = get_type(self._data)
        else:
            raise InputTypeError("请输入文本")

    def read_from_list(self, data_list):
        if isinstance(data_list, list):
            if len(data_list) > 0 and isinstance(data_list[0], list):
                self._data = number_encoder(data_list)
                self._types = get_type(self._data)
            else:
                raise InputTypeError("请输入二维列表")
        else:
            raise InputTypeError("请输入二维列表")

    def read_from_array(self, array):
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            self._data = array
            self._types = get_type(self._data)
        else:
            raise InputTypeError("请输入numpy二维数组")

    def auto_run(self, y_column=-1):
        self.y_column = y_column
        if self._data is None:
            raise EmptyInputError("请先通过AutoDataHandle.read*相关命令读取数据")

        if self._data.shape[0] < 1000:
            self._data_handled = abnormal_handle(self._data, self._types, 95, 5)
        else:
            self._data_handled = abnormal_handle(self._data, self._types, 99, 1)

        log_print("异常值处理结束")

        params = {'continuous_method': [ConMissingHandle.mean_fill,
                                        ConMissingHandle.median_fill,
                                        ConMissingHandle.sample_drop],
                  'discrete_method' : [DisMissingHandle.mode_fill,
                                       DisMissingHandle.sample_drop,
                                       DisMissingHandle.one_hot]
                  }

        nan_summary(self._data_handled, None)
        log_print("缺失值统计结束")

        log_print("缺失值处理方法寻优开始")
        best_param = grid_search(self.missing_handle_score, params, cv_times=self.cv_times, quiet=False)
        self._data_handled = missing_value_handle(self._data_handled, self._types, **best_param[0])
        log_print("缺失值处理结束")
        self._data_handled = one_hot_encoder(self._data_handled, self._types)
        log_print("独热编码结束")


    def missing_handle_score(self, continuous_method, discrete_method):
        arr = missing_value_handle(self._data_handled, self._types, continuous_method, discrete_method)
        corr_list = []
        for i in range(arr.shape[1]):
            if i != self.y_column:
                if (arr[:, i] == arr[:, i][0]).all():
                    continue
                corr_list.append(np.corrcoef(arr[:, i], arr[:, self.y_column])[0, 1])
        return np.nanmean(corr_list)



class AutoFeatureHandle(BaseAuto):
    pass


class AutoModelOpt(BaseAuto):
    pass

class AutoModelSelect(BaseAuto):
    pass


if __name__ == '__main__':
    arr =np.array([[1,2.1,3], [4, np.nan, 6], [np.nan, 8, 9], [10, 11, 12]])
    y = np.array([1,2,3, 3])
    arr = np.column_stack((arr, y.reshape(-1, 1)))
    atd = AutoDataHandle(5)
    atd.read_from_array(arr)
    atd.auto_run(-1)
    print(atd.handled_data)
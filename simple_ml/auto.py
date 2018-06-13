# -*- coding:utf-8 -*-

"""
The Kernel Module of simple_ml
"""

from simple_ml.base.base_model import *
from simple_ml.base.base_error import *
from simple_ml.base.base_enum import *
from simple_ml.data_handle import *
from simple_ml.feature_select import *
from simple_ml.logistic import *
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
    """
    网格搜索
    :param func:      目标函数，满足对应参数的输入和float类型输出
    :param params:    参数， 字典类型，key必须是func中定义的形参
    :param cv_times:  交叉验证次数
    :param increase:  True if func返回值越大越好 else func返回值越小越好
    :param quiet:     安静的进行着，否则启动log_print函数
    :return:          (最优参数dict, 最优得分)
    """
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

    def __init__(self, cv_times):
        self.cv_times = cv_times
        self._data = None
        self._types = None
        self._handled_data = None

    @property
    def origin_data(self):
        return self._data

    @property
    def handled_data(self):
        return self._handled_data

    @property
    def type_list(self):
        return self._types

    def auto_run(self, **kwargs):
        pass



class AutoDataHandle(BaseAuto):

    def __init__(self, cv_times=1):
        super(AutoDataHandle, self).__init__(cv_times)
        self.y_column = None
        self._head = None
        self._head_new_name = None

    @property
    def head_names(self):
        return self._head

    @property
    def head_new_names(self):
        if self._head is None:
            return None
        return self._head_new_name

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
        if y_column < 0:
            self.y_column = self._data.shape[1] + y_column
        else:
            self.y_column = y_column
        if self._data is None:
            raise EmptyInputError("请先通过AutoDataHandle.read*相关命令读取数据")

        if self._data.shape[0] < 1000:
            self._handled_data = abnormal_handle(self._data, self._types, 95, 5)
        else:
            self._handled_data = abnormal_handle(self._data, self._types, 99, 1)

        log_print("异常值处理结束")

        params = {'continuous_method': [ConMissingHandle.mean_fill,
                                        ConMissingHandle.median_fill,
                                        ConMissingHandle.sample_drop],
                  'discrete_method' : [DisMissingHandle.mode_fill,
                                       DisMissingHandle.sample_drop,
                                       DisMissingHandle.one_hot]
                  }

        nan_summary(self._handled_data, self._head)
        log_print("缺失值统计结束")

        log_print("缺失值处理方法寻优开始")
        best_param = grid_search(self.missing_handle_score, params, cv_times=self.cv_times, quiet=False)
        self._handled_data = missing_value_handle(self._handled_data, self._types, **best_param[0])
        log_print("缺失值处理结束")
        self._handled_data,  self._head_new_name = one_hot_encoder(self._handled_data, self._types, self._head)
        log_print("独热编码结束")


    def missing_handle_score(self, continuous_method, discrete_method):
        """
        - 一个用于网格搜索的函数，满足固定输入和输出
        - 输入值为确实样本处理方法，缺失值为处理方法的得分，我通过计算确实处理后的特征与模型标签的相关系数，
          并且用样本数对数进行惩罚得到的每个特征的平均值作为处理得分，越大越好
        :param continuous_method:    连续数据处理方法
        :param discrete_method:      离散树处理方法
        :return:                     float， 得分
        """
        arr = missing_value_handle(self._handled_data, self._types, continuous_method, discrete_method)
        corr_list = []
        for i in range(arr.shape[1]):
            if i != self.y_column:
                if (arr[:, i] == arr[:, i][0]).all():
                    continue
                # 通过样本数对数进行惩罚，防止样本数少导致的相关系数偏高
                corr_list.append(np.corrcoef(arr[:, i], arr[:, self.y_column])[0, 1] * np.log(arr.shape[0]))
        return np.nanmean(corr_list)


class AutoFeatureHandle(BaseAuto):

    def __init__(self, cv_times=5):
        super(AutoFeatureHandle, self).__init__(cv_times)
        self.y_column = None
        self._support = None
        self._new_head_name = None

    @property
    def support(self):
        """
        :return: 所选特征的id列p表
        """
        return self._support

    @property
    def get_select_head_name(self):
        if self._new_head_name is None:
            raise EmptyInputError("必须在auto_run参数中加入AutoDataHandle.head_nea_name结果")
        return [self._new_head_name[i] for i in self.support]

    def read_array(self, arr):
        if isinstance(arr, np.ndarray):
            self._data = arr
            self._types = get_type(self._data)
        else:
            raise InputTypeError("必须输入numpy二维数组，如果不是，请先利用AutoDataHandle模块进行处理")

    def auto_run(self, y_column=-1, new_head_name=None):
        self._new_head_name = new_head_name
        if self._data is None:
            raise EmptyInputError("请先运行read_array函数读取数组")
        if y_column < 0:
            self.y_column = self._data.shape[1] + y_column
        else:
            self.y_column = y_column

        percent_list = np.linspace(0.1, 1, 10)
        params = {'top_k' : [int(self._data.shape[1] * i) for i in percent_list]}
        if self._types[self.y_column] == LabelType.continuous:
            """
            标签为连续变量，此时只能选择 Filter.corr，Filter.var, Embedded.gbdt三种方法
            """
            params['select_type'] = [
                FilterType.corr,
                FilterType.var,
                EmbeddedType.GBDT
            ]
        else:
            """
            标签为离散，可以选择 Filter.chi2(确保data都大于0）， Filter.var, Embedded.lasso 三种方法
            """
            params['select_type'] = [
                FilterType.var,
                EmbeddedType.Lasso
            ]

        log_print("特征选择方法寻优开始")
        best_param, score = grid_search(self._get_feature_score, params, cv_times=self.cv_times, quiet=False)
        log_print("最优参数：%s，得分:%.4f" % (best_param, score) )
        if best_param['select_type'] in FilterType:
            model = Filter(best_param['select_type'], best_param['top_k'])
            new_train = model.fit_transform(self._data[:, np.arange(self._data.shape[1]) != self.y_column], None)
        else:
            model = Embedded(best_param['top_k'], best_param['select_type'])
            new_train = model.fit_transform(self._data[:, np.arange(self._data.shape[1]) != self.y_column], None)

        self._support = model.get_support
        self._handled_data = np.column_stack((new_train, self._data[:, self.y_column]))
        log_print("特征自动处理结束")

    def _get_feature_score(self, top_k, select_type):
        """
        获取当前特征选择方法的效果
        :param top_k:         前几个特征
        :param filter_type:   特征选择方法
        :return:              得分，回归用平均相关系数判断，二分类用logistic模型得分判断
        """
        x_train, y_train, x_test, y_test = train_test_split(
            self._data[:, np.arange(self._data.shape[1]) != self.y_column], self._data[:, self.y_column])

        if select_type in FilterType:
            model = Filter(select_type, top_k)
            new_x_train = model.fit_transform(x_train, y_train)
            new_x_test = model.transform(x_test)
        elif select_type in EmbeddedType:
            model = Embedded(top_k, select_type)
            new_x_train = model.fit_transform(x_train, y_train)
            new_x_test = model.transform(x_test)
        else:
            raise InputTypeError("特征选择方法不存在")

        if new_x_test.shape[1] == 0:
            raise ModelInputError("所有特征均和标签无关系")
        lr = LogisticRegression()
        lr.fit(new_x_train, y_train)
        return lr.score(new_x_test, y_test)


class AutoModelOpt(BaseAuto):

    __doc__ = "自动化模型最优调参"

    def __init__(self, opt_method=OptMethod.grid_search, cv_times=5):
        """
        自动模型调参，包括了网格搜索法、贝叶斯方法
        """
        super(AutoModelOpt, self).__init__(cv_times)
        self.opt_method = opt_method

    def read_array(self, arr):
        if isinstance(arr, np.ndarray):
            self._data = arr
            self._types = get_type(self._data)
        else:
            raise InputTypeError("必须输入numpy二维数组，如果不是，请先利用AutoDataHandle模块进行处理")

    def auto_run(self, model: BaseClassifier, y_column):
        """
        继承BaseAuto
        """
        if self._data is None:
            raise EmptyInputError("请先运行read_array函数读取数组")

        if y_column < 0:
            self.y_column = self._data.shape[1] + y_column
        else:
            self.y_column = y_column

        # 想好一个有机的体系之后再写
        pass



class AutoModelSelect(BaseAuto):
    pass


if __name__ == '__main__':
    # arr =np.array([[1,2.1,3], [4, np.nan, 6], [np.nan, 8, 9], [10, 11, 12]])
    # y = np.array([1,2,3, 3])
    # arr = np.column_stack((arr, y.reshape(-1, 1)))
    # atd = AutoDataHandle(5)
    # atd.read_from_array(arr)
    # atd.auto_run(-1)
    # print(atd.handled_data)

    np.random.seed(918)
    arr = np.random.rand(20, 10)
    y = np.random.choice([0, 1], 20, True)
    arr = np.column_stack((arr, y.reshape(-1, 1)))
    afh = AutoFeatureHandle(cv_times=5)
    afh.read_array(arr)
    afh.auto_run(-1)

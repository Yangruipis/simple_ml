# -*- coding:utf-8 -*-
from __future__ import division, absolute_import

import numpy as np
import os
import re
import requests
from simple_ml.data_handle import *

__all__ = [
    'get_iris',
    'get_watermelon',
    'get_wine',
    'get_moon',
    'get_circle',
    'get_hastie_10_2',
    'DataCollector',
    'load'
]


# 获取该文件的绝对路径
PATH = os.path.split(os.path.realpath(__file__))[0]


def get_iris():
    x, y = load("/data_sets/iris.txt")
    return x, np.array(y, dtype='int')


def get_watermelon():
    x, y = load("/data_sets/watermelon.txt")
    return x, np.array(y, dtype='int')


def get_wine():
    x, y = load("/data_sets/wine.txt")
    return x, np.array(y, dtype='int')


def get_moon():
    x, y = load("/data_sets/moon_200.txt")
    return x, np.array(y, dtype='int')


def get_circle():
    x, y = load("/data_sets/circle_200.txt")
    return x, np.array(y, dtype='int')


def get_hastie_10_2():
    x, y = load("/data_sets/hastie_10_2.txt")
    return x, np.array(y, dtype='int')


class DataCollector:

    def __init__(self):
        # exit_code = os.system('ping 8.8.8.8')
        # if exit_code:
        #     raise ConnectionError("无网络连接")

        self._data_content = self.get_content()
        print("受支持的数据集有：", self._data_content)

    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

    def get_content(self):

        text = requests.get(self.URL).text.replace(" ", "").replace("\n", "").replace("\t", "")
        _reg = r'alt="\[DIR\]"></td><td><ahref="(.+?)">'
        reg = re.compile(_reg)
        find = reg.findall(str(text))
        name_lst = []
        flag = False
        for i in find:
            if str(i) == "abalone/":
                flag = True
            if flag:
                name_lst.append(i[:-1])
        return name_lst

    def _download_data(self, data_name):
        data_path = PATH + "/data_sets/download/" + data_name + ".data"
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = f.read()
            return data
        else:
            url = self.URL + data_name + "/"
            text = requests.get(url, timeout=2).text
            if data_name + ".data" not in text:
                return "该数据集暂时不存在"
            data_url = url + "/" + data_name + ".data"
            data = requests.get(data_url).text
            with open(data_path, 'w') as f:
                f.write(data)
            return data

    @property
    def data_content(self):
        return self._data_content

    def fetch_origin_data(self, data_name):
        if data_name not in self._data_content:
            raise ValueError("数据集名称不正确，请通过DataCollector().data_content 确认数据集")
        return self._download_data(data_name)

    def detail_data(self, data_name):
        """
        获取数据集的描述
        :param data_name: 数据集名称，通过data_content()查看
        :return:          string
        """
        if data_name not in self._data_content:
            raise ValueError("数据集名称不正确，请通过DataCollector().data_content 确认数据集")

        url = self.URL + data_name + "/"
        text = requests.get(url, timeout=2).text
        if data_name + ".names" not in text:
            return "该数据集描述暂时不存在"
        data_url = url + "/" + data_name + ".names"
        data_detail = requests.get(data_url).text
        print(data_detail)

    def fetch_handled_data(self, data_name):
        data = self.fetch_origin_data(data_name)
        lst = read_string(data, header=False, index=False)
        arr = number_encoder(lst)
        types = get_type(arr)
        arr = abnormal_handle(arr, types)
        arr = missing_value_handle(arr, types)
        arr = one_hot_encoder(arr, types)
        return arr


def dump(x, y, path):
    with open(PATH + path, 'w') as f:
        f.write("%s,%s\n" % (x.shape[0], x.shape[1]))
        for line in x:
            f.write(",".join([str(i) for i in line]) + "\n")
        f.write(",".join([str(i) for i in y]) + "\n")


def load(path):
    x = []
    with open(PATH + path, "r") as f:
        m, n = list(map(int, f.readline().strip().split(",")))
        for i in range(m):
            x.append(list(map(float, f.readline().strip().split(","))))
        y = list(map(float, f.readline().strip().split(",")))
    return np.array(x), np.array(y)

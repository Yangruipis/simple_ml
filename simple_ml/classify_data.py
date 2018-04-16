
import numpy as np
import os
# 获取该文件的绝对路径
PATH = os.path.split(os.path.realpath(__file__))[0]

#__dir__ = ["./", "./simple_ml/", "../"]

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


def dump(x, y, path):
    with open(PATH + path, 'w') as f:
        f.write("%s,%s\n" % (x.shape[0], x.shape[1]))
        for line in x:
            f.write(",".join([str(i) for i in line]) + "\n")
        f.write(",".join([str(i) for i in y]) + "\n")


def load(path):
    x = []
    y = None
    with open(PATH + path, "r") as f:
        m, n = list(map(int, f.readline().strip().split(",")))
        for i in range(m):
          x.append(list(map(float, f.readline().strip().split(","))))
        y = list(map(float, f.readline().strip().split(",")))
    return np.array(x), np.array(y)

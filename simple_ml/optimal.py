# -*- coding:utf-8 -*-

"""
优化问题：
- 无约束最小化
- 有约束最小化
- 二次规划


优化算法
- 网格搜索
- 随机搜索
  - 模拟退火
  - 粒子群
  - 遗传算法
- 基于梯度
  - 爬山法
- 基于贝叶斯
  - 贝叶斯优化
"""

import numpy as np
from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *


class FindMin:

    def __init__(self, optimal_method=OptMethod.down_hill):
        self.optimal_method = optimal_method

    def run(self, up_bound, down_bound):
        pass


class BaseOptimal:

    def __init__(self, func, x_init, up_bound, down_bound):
        """
        基本优化器
        :param func:       目标函数，只接受一个参数x，如果不是则通过lambda转为该函数
        :param x_init:     x的初始值x_init， array
        :param up_bound:   x的上届，array
        :param down_bound: x的下届，array
        """
        if not isinstance(x_init, np.ndarray) or len(x_init.shape) != 1:
            raise InputTypeError("初始值必须为一维数组")
        self.func = func
        self.x_init = x_init
        if up_bound is None:
            self.up_bound = np.ones(len(x_init)) * np.inf
        else:
            self.up_bound = up_bound

        if down_bound is None:
            self.down_bound = - np.ones(len(x_init)) * np.inf
        else:
            self.down_bound = down_bound

    def get_value(self, x):
        return self.func(x)


class DownHill(BaseOptimal):

    __doc__ = "爬山法求解最小化问题"

    def __init__(self, func, x_init, up_bound=None, down_bound=None, iter_times=1000):
        """
        爬山法求最小值
        优点：
            - 简单
            - 快
        缺点：
            - 如果不是凸优化，则很容易陷入局部最优
        """
        super(DownHill, self).__init__(func, x_init, up_bound, down_bound)
        self.iter_times = iter_times

    def _gen_solution(self, x, percent=0.1):
        _min = np.inf
        best_x = None
        for i in range(len(x)):
            temp = x[i]
            delta = temp * percent if temp != 0 else 1
            x[i] = min(self.up_bound[i], temp + delta)
            sol1 = self.get_value(x)
            if sol1 < _min:
                _min = sol1
                best_x = x.copy()
            x[i] = max(self.down_bound[i], temp - delta)
            sol2 = self.get_value(x)
            if sol2 < _min:
                _min = sol2
                best_x = x.copy()
        return best_x, _min

    def run(self):
        x = self._gen_solution(self.x_init)[0]
        for i in range(self.iter_times):
            x = self._gen_solution(x)[0]
        return x


class SimulatedAnneal(BaseOptimal):

    def __init__(self, func, x_init, up_bound=None, down_bound=None, iter_times=1000,
                 t0=1e10, t_min=1e-8, delta=0.9):
        super(SimulatedAnneal, self).__init__(func, x_init, up_bound, down_bound)
        self.iter_times = iter_times
        self.t0 = t0
        self.t_min = t_min
        self.delta = delta

    def get_new_x(self, x, T):
        x_after = x + (np.random.rand(len(x)) * 2 - 1) * T
        x_after = np.array(list(map(lambda i, j: min(i, j), self.up_bound, x_after)))
        x_after = np.array(list(map(lambda i, j: max(i, j), self.down_bound, x_after)))
        return x_after

    def run(self):
        f = self.func(self.x_init)
        x = self.x_init.copy()
        t = self.t0
        while t > self.t_min:
            for i in range(self.iter_times):
                new_x = self.get_new_x(x, t)
                f_x = self.func(new_x)
                delta__e = f_x - f
                #
                if delta__e < 0:
                    f = f_x
                    x = new_x
                    break
                else:
                    # p_k = 1.0 / (1 + np.exp(- delta_E / self.func(T)))
                    p_k = np.exp(- delta__e / t)
                    if np.random.random() < p_k:
                        f = f_x
                        x = new_x
                        break
            t *= self.delta
        return x


class PSO(BaseOptimal):

    def __init__(self, func, x_init, up_bound=None, down_bound=None, iter_times=1000):
        super(PSO, self).__init__(func, x_init, up_bound, down_bound)
        self.iter_times = iter_times

    def run(self):
        pass



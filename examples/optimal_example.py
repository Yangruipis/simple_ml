# -*- coding:utf-8 -*-

from simple_ml.optimal import *


def down_hill_example():
    dh = DownHill(lambda x: (x - 1)**2 + 1, np.array([-1]))
    print(dh.run())


def sa_example():
    sa = SimulatedAnneal(lambda x: (x - 1)**2 + 1, np.array([-1]))
    print(sa.run())


def method_compare():
    import scipy.optimize as so
    so.fmin()


if __name__ == '__main__':
    down_hill_example()
    sa_example()


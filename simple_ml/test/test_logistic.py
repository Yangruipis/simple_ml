# -*- coding:utf-8 -*-

import unittest
import numpy as np
from simple_ml.logistic import LogisticRegression
from simple_ml.base.base_error import *


class TestLogistic(unittest.TestCase):

    def test_type_check(self):
        x = [[1, 2], [2, 4], [5, 6]]
        y = [1, 0, 0]
        lr = LogisticRegression()
        self.assertRaises(FeatureTypeError, lr.fit, x, y)
        x = np.array(x)
        y = np.array(y+[1])
        self.assertRaises(SampleNumberMismatchError, lr.fit, x, y)

    def test_add_ones(self):
        pass

    def test_gradient(self):
        pass

    def test_loss_function(self):
        pass

    def test_weight(self):
        pass

    def test_predict(self):
        pass


if __name__ == "__main__":
    unittest.main()


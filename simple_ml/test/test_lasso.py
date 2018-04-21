# -*- coding:utf-8 -*-

import unittest
import numpy as np
from simple_ml.logistic import Lasso
from simple_ml.base.base_error import *


class TestLasso(unittest.TestCase):

    def test_w_update(self):
        x = np.array([[1, 0],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
        y = np.array([1, 1, 0, 0])
        lasso = Lasso(lamb=0.1)
        lasso.fit(x, y)
        self.assertNotEqual(0, lasso.w.any())
        self.assertEqual(0, min(lasso.w))

    def test_loss_func(self):
        lasso = Lasso()
        lasso.x = np.array([[1, 0], [0, 1]])
        lasso.y = np.array([0, 1])
        self.assertEqual(lasso._loss_function_value(np.array([0, 0])), 0.25)

    def test_error(self):
        x = np.array([1,0,0,1])
        y = np.array([1,0])
        lasso = Lasso()
        self.assertRaises(FeatureTypeError, lasso.fit, x, y)

        x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
        y = np.array([0.1, 0.2, 0.3, 0.4])
        lasso = Lasso()
        self.assertRaises(LabelTypeError, lasso.fit, x, y)

if __name__ == '__main__':
    unittest.main()
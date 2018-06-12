# -*- coding:utf-8 -*-

from simple_ml.base.base_error import *
from simple_ml.regression import MultiRegression
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class TestMultiRegression(unittest.TestCase):

    def test_if_no_freedom(self):
        x_train = np.array([[1, 2, 3],
                           [2, 1, 2]])
        y_train = np.array([1.1, 1.2])
        reg = MultiRegression()
        self.assertRaises(ParamInputError, reg.fit, x_train, y_train)

    def test_if_singular(self):
        x_train = np.array([[0, 0],
                            [1, 1]])
        y_train = np.array([1.0, 2.0])
        reg = MultiRegression()
        self.assertEqual(np.linalg.det(np.dot(x_train.T, x_train)), 0)
        self.assertRaises(ParamInputError, reg.fit, x_train, y_train)

    def test_result(self):
        x_train = np.array([[1], [2]])
        y_train = np.array([1, 2])
        reg = MultiRegression()
        reg.fit(x_train, y_train)
        self.assertAlmostEqual(reg.beta[0], 1.0)
        self.assertAlmostEqual(reg.r_square, 1.0)
        assert_array_almost_equal(reg.beta, [1.0])


if __name__ == '__main__':
    unittest.main()

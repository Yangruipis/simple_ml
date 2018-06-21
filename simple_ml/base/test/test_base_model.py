# -*- coding:utf-8 -*-

import unittest
from simple_ml.base.base_model import *
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from simple_ml.base.base_error import *



class TestBaseModel(unittest.TestCase):

    def test_base_model_transfer(self):

        base_model = BaseModel()
        x = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0]
                      ])
        y = np.array([-1, -1, 1, -1, -1])
        base_model._init(x, y)
        assert_array_equal(base_model.y, y + 1)
        y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        base_model._clear()
        base_model._init(x, y)
        assert_array_equal(base_model.y, y)

    def test_input_nan_or_inf(self):

        x = np.array([[1, 2, np.nan],
                     [3, 4.0, 5],
                     [1, 1, 1]])
        y = np.array([1, 2, 3])
        base_model = BaseModel()
        self.assertRaises(ArrayContainNANorINF,base_model._init, x, y)

        x = np.array([[1, 2, np.inf],
                      [3, 4.0, 5],
                      [1, 1, 1]])
        y = np.array([1, 2, 3])
        base_model = BaseModel()
        self.assertRaises(ArrayContainNANorINF, base_model._init, x, y)

    def test_input_not_np_array(self):
        x = [[1, 2, 3], [4, 5, 6]]
        y = np.array([1, 0])
        base_model = BaseModel()
        self.assertRaises(FeatureTypeError, base_model._init, x, y)


if __name__ == '__main__':
    unittest.main()
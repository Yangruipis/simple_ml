# -*- coding:utf-8 -*-

import unittest
from simple_ml.classify_data import DataCollector
import numpy as np

class TestDataCollector(unittest.TestCase):

    def test_get_iris(self):
        dc = DataCollector()
        x = dc.fetch_handled_data("iris")
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape[0], 150)
        self.assertEqual(x.shape[1], 6)

if __name__ == '__main__':
    unittest.main()

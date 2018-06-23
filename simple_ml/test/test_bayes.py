# -*- coding:utf-8 -*-

import unittest
from simple_ml.bayes import *
import numpy as np


class TestBayes(unittest.TestCase):

    def test_get_normal_prob(self):
        prob = NaiveBayes._get_normal_prob(0, 0, 1)
        self.assertAlmostEqual(np.round(prob, 1), 0.4)

        prob = NaiveBayes._get_normal_prob(0, 918, 9188)
        self.assertAlmostEqual(np.round(prob, 1), 0.0)


if __name__ == '__main__':
    unittest.main()

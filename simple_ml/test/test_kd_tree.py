import unittest
from simple_ml.kd_tree import KDTree


class TestKDTree(unittest.TestCase):
    def test_search(self):
        kd_tree = KDTree(k=2)
        points = [
            [3, 6], [17, 15], [13, 15], [6, 12],
            [9, 1], [2, 7], [10, 19]
        ]

        for point in points:
            kd_tree.insert(point)

        point1 = [10, 19]
        self.assertEqual(kd_tree.search(point1), True)

        point2 = [12, 19]
        self.assertEqual(kd_tree.search(point2), False)


if __name__ == '__main__':
    unittest.main()

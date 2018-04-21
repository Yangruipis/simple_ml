# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

import numpy as np
from numpy.linalg import norm

from simple_ml.base.base_enum import DisType
from simple_ml.base.base_error import *


__all__ = ['KMeans', 'Hierarchical', 'DisType']


class KMeans(object):

    def __init__(self, k, dis_type=DisType.Eculidean, d=1):
        self.k = k
        self.disType = dis_type
        self.d = d

    def _get_dis_func(self, x1, x2):

        if self.disType == DisType.Eculidean:
            return norm(x1 - x2)
        elif self.disType == DisType.Manhattan:
            return norm(x1 - x2, 1)
        elif self.disType == DisType.Minkowski:
            return norm(x1 - x2, self.d)
        elif self.disType == DisType.Chebyshev:
            return norm(x1 - x2, np.inf)
        elif self.disType == DisType.CosSim:
            return np.dot(x1, x2) / (norm(x1) * norm(x2))
        else:
            raise DistanceTypeError

    def _clear(self):
        self.variable_num = self.sample_num = 0
        self.x = None

    def _init(self, x):
        self._clear()
        self.variable_num = x.shape[1]
        self.sample_num = x.shape[0]
        self.x = x
        self._labels = np.zeros(self.sample_num)

    def fit(self, x):
        self._init(x)
        self._fit()

    @staticmethod
    def _center_is_changed(center, updated_center):
        for i in range(center.shape[0]):
            if not np.array_equal(center[i], updated_center[i]):
                return True
        return False

    def _judge_single_sample(self, idx, x, center_point):
        distances = list(map(lambda i: self._get_dis_func(x, i), center_point))
        if self.disType == DisType.CosSim:
            self._labels[idx] = np.argmax(distances)
        else:
            self._labels[idx] = np.argmin(distances)

    def _get_new_center(self, old_center):
        new_center = old_center.copy()
        for i, label in enumerate(np.unique(self.labels)):
            new_center[i] = np.mean(self.x[self._labels == label, :], axis=0)
        return new_center

    def _fit(self):
        init_point_id = np.random.choice(np.arange(self.sample_num), self.k)

        center_point = self.x[init_point_id, :]
        center_update_point = center_point + 1

        while self._center_is_changed(center_point, center_update_point):
            center_update_point = center_point
            for i, x in enumerate(self.x):
                self._judge_single_sample(i, x, center_point)
            # print(self._labels)
            center_point = self._get_new_center(center_point)

    @property
    def labels(self):
        return self._labels


class ClusterNode(object):

    def __init__(self, left, right, parent, ids, distance=0):
        self.left = left
        self.right = right
        self.parent = parent
        self.ids = ids        # 当前节点的所有样本编号
        self.distance = distance


class Hierarchical(object):

    def __init__(self, dis_type=DisType.Eculidean, d=1):
        self.disType = dis_type
        self.d = d

    def _get_dis_func(self, x1, x2):

        if self.disType == DisType.Eculidean:
            return norm(x1 - x2)
        elif self.disType == DisType.Manhattan:
            return norm(x1 - x2, 1)
        elif self.disType == DisType.Minkowski:
            return norm(x1 - x2, self.d)
        elif self.disType == DisType.Chebyshev:
            return norm(x1 - x2, np.inf)
        elif self.disType == DisType.CosSim:
            return - np.dot(x1, x2) / (norm(x1) * norm(x2))
        else:
            raise DistanceTypeError

    def _clear(self):
        self.variable_num = self.sample_num = 0
        self.x = None
        self.disMat = None
        self.root = None

    def _init(self, x):
        self._clear()
        self.variable_num = x.shape[1]
        self.sample_num = x.shape[0]
        self.x = x
        self._labels = np.arange(self.sample_num)

    def fit(self, x):
        self._init(x)
        self.root = self._fit()

    def _cal_dis_mat(self):
        self.disMat = np.ones((self.sample_num, self.sample_num))
        for i in range(self.sample_num):
            for j in range(i+1, self.sample_num):
                self.disMat[i, j] = self._get_dis_func(self.x[i], self.x[j])
                self.disMat[j, i] = self.disMat[i, j]
            self.disMat[i, i] = 0.0

    def _dis_of_cluster(self, cluster_node_1: ClusterNode, cluster_node_2: ClusterNode):
        _sum = 0.0
        count = 0.0
        for i in cluster_node_1.ids:
            for j in cluster_node_2.ids:
                _sum += self.disMat[i, j]
                count += 1
        return _sum / count

    def _get_nearest_id(self, candidate_list):
        min_dis = np.inf
        nearest_pair = (-1, -1)
        for i, _ in enumerate(candidate_list):
            for j in range(i+1, len(candidate_list)):
                dis = self._dis_of_cluster(candidate_list[i], candidate_list[j])
                if dis < min_dis:
                    min_dis = dis
                    nearest_pair = (i, j)

        return nearest_pair, min_dis

    def _fit(self):

        # 1. 先计算出每个样本间的距离矩阵
        self._cal_dis_mat()

        # 2. 每个样本看成一个簇， 每次循环把最接近的簇变成同一个簇，从下往上构成一个二叉树
        candidate_list = [ClusterNode(None, None, None, [i]) for i in range(self.sample_num)]
        while True:
            if len(candidate_list) == 1:
                break
            pair, dis = self._get_nearest_id(candidate_list)

            # 更新candidate_list
            left_node = candidate_list[pair[0]]
            right_node = candidate_list[pair[1]]
            new_cluster = ClusterNode(left_node, right_node, None, None, dis)
            new_cluster.ids = left_node.ids + right_node.ids
            left_node.parent = new_cluster
            right_node.parent = new_cluster

            candidate_list.remove(left_node)
            candidate_list.remove(right_node)
            candidate_list.append(new_cluster)

        return candidate_list[0]

    def cluster(self, min_sim=None):
        if min_sim is None:
            min_sim = self.max_dis
        clusters = []
        self._recursion(self.root, min_sim, clusters)
        for i, cluster in enumerate(clusters):
            for j in cluster:
                self._labels[j] = i
        return self._labels

    @staticmethod
    def _recursion(node, threshold, clusters_list):
        if node.left is None or node.right is None:
            clusters_list.append(node.ids)
            return

        if node.distance < threshold:
            # 此时跳出，并且记录当前的节点
            clusters_list.append(node.ids)
            return

        Hierarchical._recursion(node.left, threshold, clusters_list)
        Hierarchical._recursion(node.right, threshold, clusters_list)

    @property
    def labels(self):
        return self._labels

    @property
    def max_dis(self):
        return self.root.distance

if __name__ == '__main__':
    km = KMeans(10)
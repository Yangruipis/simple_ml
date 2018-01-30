# -*- coding:utf-8 -*-


from .my_classifier import *
from .my_enumrate import DisType
from .score import *
from .classify_plot import classify_plot
from .my_error import DistanceTypeError


class MyKnn(MyClassifier):

    def __init__(self, k=1, distance_type=DisType.Eculidean):
        super(MyKnn, self).__init__()
        self.k = k
        self.dist_type = distance_type
        self.x = None
        self.y = None

    def fit(self, x, y):
        self._init(x, y)
        self._fit(x, y)

    def _fit(self, x, y):
        """
        由于是惰性学习，此处无需做任何事
        """
        pass

    def predict(self, x, k=None, dist_type=None):
        if self.x is None:
            raise ModelNotFittedError
        if k is None:
            k = self.k
        else:
            k = k
        if dist_type is None:
            dist_type = self.dist_type
        dist_func = self._get_dist_func(dist_type)
        return list(map(lambda i: self._predict_single_sample(i, k, dist_func), x))

    def _predict_single_sample(self, x, k, dist_func):
        sim_list = list(map(lambda i: dist_func(x, i), self.x))
        sim_y_list = zip(sim_list, self.y)
        selected_sim_y = sorted(sim_y_list, key=lambda i: i[0], reverse=False)[:k]
        return self._vote([i[1] for i in selected_sim_y])

    @staticmethod
    def _vote(y_list):
        count_dict = dict(Counter(y_list))
        return max(count_dict, key=count_dict.get)

    @staticmethod
    def _get_dist_func(dist_type):

        if dist_type == DisType.Eculidean:
            # 向量相减的2范数，就是马氏距离
            return lambda x1, x2: np.linalg.norm(x1-x2, 2)
        elif dist_type == DisType.Manhattan:
            return lambda x1, x2: np.linalg.norm(x1-x2, 1)
        elif dist_type == DisType.Chebyshev:
            return lambda x1, x2: np.linalg.norm(x1-x2, np.inf)
        elif dist_type == DisType.CosSim:
            return lambda x1, x2: -np.dot(x1, x2) / (np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2))
        else:
            raise DistanceTypeError

    def score(self, x, y, k=None, dist_type=None):
        y_predict = self.predict(x, k, dist_type)
        count_dict = dict(Counter(y_predict))
        if len(count_dict) <= 2:
            # binary classifyn
            f1_score = classify_f1(y_predict, y)
        else:
            f1_score = classify_f1_macro(y_predict, y)
        return f1_score

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title='My kNN')


class Node:

    __slot__ = ['left', 'right', 'parent', 'value', 'dimension', 'sampleIds']

    def __init__(self, left, right, parent, value, dimension, sample_ids):
        self.left = left
        self.right = right
        self.value = value
        self.parent = parent
        self.dimension = dimension
        self.sample_ids = sample_ids


class MyKDTree(MyKnn):

    def __init__(self, k=5, dist_type=DisType.Eculidean):
        super(MyKDTree, self).__init__(k, dist_type)

    @staticmethod
    def _choose_split_feature(x, ids):
        x = x[ids]
        variance_list = list(map(lambda i: np.var(i), x.T))
        split_feature_id = variance_list.index(max(variance_list))
        median = np.median(x.T[split_feature_id])
        # split_node = node(None, None, median, split_feature_id, ids)
        return split_feature_id, median

    def _fit(self, x, y):
        """
        此处建KD树
        """
        self.root_node = Node(None, None, None, None, None, np.arange(self.x.shape[0]))
        self._build_kd_tree(self.root_node, 0)

    def _build_kd_tree(self, input_node: Node, depth):
        if len(input_node.sample_ids) == 0:
            return None

        sample_ids = input_node.sample_ids
        feat_id, median = self._choose_split_feature(self.x, input_node.sample_ids)
        input_node.value = median     # sample_ids[self.x[sample_ids, feat_id]==median]
        input_node.dimension = feat_id
        left_ids = sample_ids[self.x[sample_ids, feat_id] < median]
        left_node = Node(None, None, input_node, None, None, left_ids)
        right_ids = sample_ids[self.x[sample_ids, feat_id] > median]
        right_node = Node(None, None, input_node, None, None, right_ids)
        input_node.left = self._build_kd_tree(left_node, depth+1)
        input_node.right = self._build_kd_tree(right_node, depth+1)
        return input_node

    def _predict_single_sample(self, x, k, dist_func):
        """
        1 . 从root节点开始，DFS搜索直到叶子节点，同时在stack中顺序存储已经访问的节点。
        2. 如果搜索到叶子节点，当前的叶子节点被设为最近邻节点。
        3. 然后通过stack回溯:
        4. 如果当前点的距离比最近邻点距离近，更新最近邻节点.
        5. 然后检查以最近距离为半径的圆是否和父节点的超平面相交.
        6. 如果相交，则必须到父节点的另外一侧，用同样的DFS搜索法，开始检查最近邻节点。
        7. 如果不相交，则继续往上回溯，而父节点的另一侧子节点都被淘汰，不再考虑的范围中.
        8. 当搜索回到root节点时，搜索完成，得到最近邻节点。
        """
        pass

    def _search_kd_tree(self, x, the_node: Node, dist_func):
        if x[the_node.dimension] == the_node.value:
            return the_node

        if the_node.left is None and the_node.right is None:
            return the_node

        if x[the_node.dimension] < the_node.value:
            return self._search_kd_tree(x, the_node.left, dist_func)
        else:
            return self._search_kd_tree(x, the_node.right, dist_func)

    def _back_trace(self, x, the_node: Node, best_node: Node):
        """
        如果x和node的父节点不相交，则一直往上回溯到近邻节点
        如果x和node的父节点相交，则要回溯父节点的另一分支
        # TODO
        """
        pass

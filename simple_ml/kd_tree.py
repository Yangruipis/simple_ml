"""
Reference: https://www.geeksforgeeks.org/k-dimensional-tree/
"""
from typing import List


class KDTree:
    def __init__(self, k: int):
        self.k = k
        self.root = None

    def insert(self, point: List[int]):
        self.root = insert(self.k, self.root, point)

    def search(self, point: List[int]):
        return search(self.k, self.root, point)


class KDNode:
    def __init__(self, k: int, left, right):
        self.point = [0] * k
        self.left = left
        self.right = right


def new_node(k: int, arr: List[int]) -> KDNode:
    temp = KDNode(k, None, None)
    for i in range(k):
        temp.point[i] = arr[i]
    return temp


def insert_rec(k: int, root: KDNode, point: List[int], depth: int) -> KDNode:
    if root == None:
        return new_node(k, point)

    cd = depth % k

    if point[cd] < root.point[cd]:
        root.left = insert_rec(k, root.left, point, depth + 1)
    else:
        root.right = insert_rec(k, root.right, point, depth + 1)
    return root


def insert(k: int, root: KDNode, point: List[int]) -> KDNode:
    return insert_rec(k, root, point, 0)


def are_points_same(k: int, point1: List[int], point2: List[int]) -> bool:
    for i in range(k):
        if point1[i] != point2[i]:
            return False
    return True


def search_rec(k: int, root: KDNode, point: List[int], depth: int) -> bool:
    if root == None:
        return False
    if are_points_same(k, root.point, point):
        return True

    cd = depth % k
    if point[cd] < root.point[cd]:
        return search_rec(k, root.left, point, depth + 1)
    return search_rec(k, root.right, point, depth + 1)


def search(k: int, root: KDNode, point: List[int]) -> bool:
    return search_rec(k, root, point, 0)
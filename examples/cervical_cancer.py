# -*- coding:utf-8 -*-

"""
宫颈癌数据，来自：
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
"""

from simple_ml.auto import AutoDataHandle

PATH = './risk_factors_cervical_cancer.csv'

adh = AutoDataHandle()
adh.read_from_file(PATH, header=True, index=False, sep=',')

adh.auto_run(-1)
for i, head in enumerate(adh._head):
    print(i, head, adh.type_list[i])




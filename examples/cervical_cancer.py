# -*- coding:utf-8 -*-

"""
宫颈癌数据，来自：
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
"""

from simple_ml.auto import AutoDataHandle, AutoFeatureHandle

PATH = './risk_factors_cervical_cancer.csv'

adh = AutoDataHandle()
adh.read_from_file(PATH, header=True, index=False, sep=',')

adh.auto_run(-1)
arr = adh.handled_data

afh = AutoFeatureHandle(5)
afh.read_array(arr)

afh.auto_run(-1, adh.head_new_names)
arr = afh.handled_data

print(arr.shape)
print(afh.get_select_head_name)

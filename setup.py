# -*- coding:utf-8 -*-

from setuptools import setup
import sys

"""
setuptools比distutils.core多了检查包依赖功能
"""


if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')


setup(
    name="simple_ml",
    version="1.0",
    author="Ray Yang",
    author_email="yangruipis@163.com",
    description=("A simple machine learning algorithm implementation"),
    license="MIT",

    url="https://github.com/Yangruipis/simple_ml",
    packages=['simple_ml', 'simple_ml.base'],
    install_requires=['enum34 >1.0.0', 'numpy>=1.10', 'setuptools>=16.0', 'matplotlib>2.0.0', 'scipy>0.15.0', 'requests>2.10'],
    # dependency_links = ['https://github.com/minepy/minepy/archive/1.2.2.tar.gz#egg=minepy-1.2.2']
)

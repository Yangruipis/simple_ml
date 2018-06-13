# -*- coding:utf-8 -*-

"""
setuptools比distutils.core多了检查包依赖功能
"""
import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

    def _find_packages(where='.', exclude=()):
        """Return a list all Python packages found within directory 'where'
        'where' should be supplied as a "cross-platform" (i.e. URL-style)
        path; it will be converted to the appropriate local path syntax.
        'exclude' is a sequence of package names to exclude; '*' can be used
        as a wildcard in the names, such that 'foo.*' will exclude all
        subpackages of 'foo' (but not 'foo' itself).
        """
        out = []
        stack = [(convert_path(where), '')]
        while stack:
            where, prefix = stack.pop(0)
            for name in os.listdir(where):
                fn = os.path.join(where, name)
                if ('.' not in name and os.path.isdir(fn) and
                        os.path.isfile(os.path.join(fn, '__init__.py'))):
                    out.append(prefix+name)
                    stack.append((fn, prefix + name + '.'))
        for pat in list(exclude)+['ez_setup', 'distribute_setup']:
            from fnmatch import fnmatchcase
            out = [item for item in out if not fnmatchcase(item, pat)]


def read(fname):
    with open(fname, 'r', encoding='UTF-8') as fp:
        content = fp.read()
    return content


PUBLISH_CMD = 'python setup.py register sdist upload'


if sys.version_info < (3,5):
    sys.exit('Sorry, Python < 3.5 is not supported')


setup(
    name="simple_male",
    version="0.1.1",
    author="Ray Yang",
    author_email="yangruipis@163.com",
    description=("A machine learning algorithm implementation"),
    license="MIT",
    url="https://yangruipis.github.io/simple_ml/",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    # packages=find_packages(exclude=("test*")),
    packages=['simple_ml', 'simple_ml.base', 'simple_ml.data_sets'],
    package_data={'': ['*.md', '*.txt', '*.data']},
    include_package_data=True,
    install_requires=['numpy>=1.10','openopt>0.5', 'setuptools>=16.0', 'matplotlib>2.0.0', 'scipy>0.15.0', 'requests>2.10'],
    # dependency_links = ['https://github.com/minepy/minepy/archive/1.2.2.tar.gz#egg=minepy-1.2.2']
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='xbert',
    version='0.1.0',
    description='universal xbert frame by tf2',
    long_description='xbert: https://github.com/xuyingjie521/xbert',
    license='GNU General Public License 3.0',
    url='https://github.com/xuyingjie521/xbert',
    author='yuyangmu',
    author_email='1812316597@163.com',
    install_requires=['tensorflow>=2.2.0'],
    packages=find_packages()
)

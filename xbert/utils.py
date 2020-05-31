# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/31

# the function of this file: implement various auxiliary functions


import sys
import six

is_py2 = six.PY2

__all__ = ['is_sting', ]


if not is_py2:
    basestring = str


def is_sting(s):
    """
    字符串判断函数
    :param s: 字符串
    :return:  Boolean
    """
    return isinstance(s, basestring)


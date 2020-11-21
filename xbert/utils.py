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
import numpy as np

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


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/24

# the function of this file: Provide custom decorator functions and Layer for tf.keras.layer based on tf2


from xbert.backend import keras


class Layer(keras.layers.Layer):
    """
    The custom layer of my project can be masked
    """
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True


def integerize_shape(func):
    """确保input_shape一定是int或None的自定义装饰器
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func




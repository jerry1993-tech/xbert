# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/24

# the function of this file: Provide custom bias layer based on tf2

from .custom_decorator import Layer, integerize_shape
from xbert.backend import K


class BiasAdd(Layer):
    """自定义bias层可辅助添加
    """
    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)




# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/31

# the function of this file: Provide custom feed forward layer based on tf2


from tensorflow.keras import initializers, activations
from .custom_decorator import Layer, integerize_shape
from tensorflow.keras.layers import Dense


class FeedForward(Layer):
    """
    用两个dense层实现feed forward层
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        self.dense_1 = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.dense_2 = Dense(units=output_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = inputs
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
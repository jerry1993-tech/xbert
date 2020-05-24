# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/31

# the function of this file: Provide custom normalization layer based on tf2


from tensorflow.keras import initializers, activations
from tensorflow.keras.layers import Dense
from xbert.backend import K
from .custom_decorator import Layer


class BasicLayerNormalization(Layer):
    """
    实现Basic Layer Normalization 层
    """
    def __init__(self, **kwargs):
        super(BasicLayerNormalization, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()   # 初始化一个很小的数 1e-14

    def build(self, input_shape):
        super(BasicLayerNormalization, self).build(input_shape)
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(shape=shape,
                                     initializer="ones",
                                     name="gamma")
        self.beta = self.add_weight(shape=shape,
                                    initializer="zeros",
                                    name="beta")

    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)     # 增加一个很小的数避免开根号求导梯度出现问题
        outputs = (inputs - mean) / std
        outputs *= self.gamma
        outputs += self.beta
        return outputs

    def get_config(self):                         # 增加 get_config 方法使得 tf.keras 模型能够保存为 h5
        config = {"epsilon": self.epsilon}
        base_config = super(BasicLayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """
    实现Conditional Layer Normalization 层
    当参数hidden_*仅为有输入条件时使用，即conditional=True
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        initializer='zeros',
                                        name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         initializer='ones',
                                         name='gamma')

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)

            if self.center:
                self.beta_dense = Dense(units=shape[0],
                                        use_bias=False,
                                        kernel_initializer='zeros')
            if self.scale:
                self.gamma_dense = Dense(units=shape[0],
                                         use_bias=False,
                                         kernel_initializer='zeros')

    def call(self, inputs):
        """
        如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


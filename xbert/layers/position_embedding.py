# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/31

# the function of this file: Provide custom position embedding layer based on tf2


from tensorflow.keras import initializers, activations
from xbert.backend import K
from .custom_decorator import Layer
import tensorflow as tf


class PositionEmbedding(Layer):
    """实现 Position Embedding，且Embedding可训练。
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(self.input_dim, self.output_dim),
                                          initializer=self.embeddings_initializer
                                          )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if K.dtype(position_ids) != 'int32':
                position_ids = K.cast(position_ids, 'int32')
            pos_embeddings = K.gather(self.embeddings, position_ids)
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            pos_embeddings = self.embeddings[:seq_len]
            pos_embeddings = K.expand_dims(pos_embeddings, 0)
            if self.merge_mode != 'add':
                pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinCosPositionEmbedding(Layer):
    """
    实现 Google 论文中的 Sin-Cos 形式的位置 Embedding 层
    """
    def __init__(self, v_dim, merge_mode="add", **kwargs):
        """
        Args:
            v_dim: embedding 的维度
            merge_mode: 与位置 embedding 的合并模式， "add" 表示相加，"concate" 表示拼接
            **kwargs:
        """
        super(SinCosPositionEmbedding, self).__init__(**kwargs)
        self.v_dim = v_dim
        self.merge_mode = merge_mode

    def call(self, inputs):
        pid = tf.range(K.shape(inputs)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(inputs)[0], 1])
        pv = self.idx2pos(pid)
        if self.merge_mode == "add":
            return pv + inputs
        else:
            return K.concatenate([inputs, pv])

    def idx2pos(self, pid):
        pid = K.cast(pid, "float32")
        pid = K.expand_dims(pid, 2)
        pj = 1. / K.pow(10000., 2. / self.v_dim * K.arange(self.v_dim // 2,
                                                           dtype="float32"))
        pj = K.expand_dims(pj, 0)
        pv = K.dot(pid, pj)
        pv1, pv2 = K.sin(pv), K.cos(pv)
        pv1, pv2 = K.expand_dims(pv1, 3), K.expand_dims(pv2, 3)
        pv = K.concatenate([pv1, pv2], 3)
        return K.reshape(pv, (K.shape(pv)[0], K.shape(pv)[1], self.v_dim))

    def compute_output_shape(self, input_shape):
        if self.merge_mode == "add":
            return input_shape
        else:
            return input_shape[:-1] + (input_shape[-1] + self.vdim)

    def get_config(self):
        config = {"v_dim": self.v_dim, "merge_mode": self.merge_mode}
        base_config = super(SinCosPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

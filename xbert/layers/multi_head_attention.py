# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/24

# the function of this file: Provide custom Multiple Head Attention layer based on tf2

import tensorflow as tf
from tensorflow.keras import initializers, activations
from tensorflow.keras.layers import *
from xbert.backend import K, sequence_masking
from .custom_decorator import Layer


class MultiHeadAttention(Layer):
    """实现多头注意力层
    """
    def __init__(
        self,
        num_heads,
        head_size,
        key_size=None,
        use_bias=True,
        attention_scale=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        """
        Args:
            num_heads: 多头 attention 的数量
            head_size: 每个 attention head 的维度
            key_size: 输出的 Embedding 维度
            **kwargs:
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = num_heads
        self.head_size = head_size
        self.out_dim = num_heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, a_mask=None, p_bias=None):
        """
        实现多头注意力机制
        Args:
            inputs: 输入是一个 list，[q, k, v, mask, a_mask, p_bias]，mask 可少，q,k,v 不可少。
            q_mask: 对输入的query序列的mask，主要是将输出结果的padding部分置0。
            v_mask: 对输入的value序列的mask，主要是防止attention读取到padding信息。
            a_mask: 对attention矩阵的mask，不同的attention mask对应不同的应用。
            p_bias: 在attention里的位置偏置。一般用来指定相对位置编码的种类。
            **kwargs:
        Returns:返回经过注意力机制的结果
        """
        q, k, v = inputs[:3]
        q_mask, v_mask, n = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        if a_mask:
            a_mask = inputs[n]
            n += 1
        # 线性变换
        qw = self.q_dense(q)   # [batch_size, seq_len, num_heads * key_size]
        kw = self.k_dense(k)   # [batch_size, seq_len, num_heads * key_size]
        vw = self.v_dense(v)   # [batch_size, seq_len, num_heads * head_size]

        # 形状变换
        # [batch_size, seq_len, num_heads, key_size]
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        # [batch_size, seq_len, num_heads, key_size]
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        # [batch_size, seq_len, num_heads, head_size]
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))

        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            pos_embeddings = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, pos_embeddings)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        a = sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)

        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', a, pos_embeddings)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        # 返回结果
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/24

# the function of this file: Provide custom embedding layer based on tf2


from xbert.backend import keras, K


class Embedding(keras.layers.Embedding):
    """Custom Embedding Layer
    """
    def compute_mask(self, inputs, mask=None):
        """first token is not masked
        """
        if self._current_mode == 'embedding':
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """define mode parameter:
        if 'embedding': Common Embedding layer
        elif 'dense'  : Dense layer without bias
        """
        self._current_mode = mode
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        if self._current_mode == 'embedding':
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (K.int_shape(self.embeddings)[0],)

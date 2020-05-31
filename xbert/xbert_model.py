# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @group xuyongfu. All Rights Reserved.
#
# @author: yuyangmu
# @date: 2020/05/24

# the function of this file: implement the main structure of transformer and bert based on tf2


from .backend import keras, K

from .layers.embedding_layer import Embedding
from .layers.bias_add import BiasAdd
from .layers.position_embedding import PositionEmbedding
from .layers.multi_head_attention import MultiHeadAttention
from .layers.layer_normalization import BasicLayerNormalization, LayerNormalization
from .layers.feed_forward import FeedForward
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import json
from utils import is_sting


__all__ = [
    'get_custom_objects', 'set_custom_objects', 'build_xbert_model',
]


def get_custom_objects() -> dict:
    return {
        'Embedding': Embedding,
        'BiasAdd': BiasAdd,
        'MultiHeadAttention': MultiHeadAttention,
        'BasicLayerNormalization': BasicLayerNormalization,
        'LayerNormalization': LayerNormalization,
        'PositionEmbedding': PositionEmbedding,
        'FeedForward': FeedForward,
    }


def set_custom_objects() -> None:
    for key, val in get_custom_objects().items():
        keras.utils.get_custom_objects()[key] = val


class TransformerBlock(object):
    """定义xbert模型基类TransformerBlock
    """
    def __init__(
        self,
        vocab_size,               # 词表大小
        hidden_size,              # 编码维度
        num_hidden_layers,        # Transformer总层数
        num_attention_heads,      # Attention的头数
        intermediate_size,        # FeedForward的隐层维度
        hidden_act,               # FeedForward隐层的激活函数
        dropout_rate=None,        # Dropout比例
        embedding_size=None,      # 是否指定embedding_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,     # 是否固定序列长度
        keep_tokens=None,         # 要保留的词ID列表
        layers=None,              # 外部传入的Keras层
        name=None,                # 模型名称
        **kwargs
    ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_tokens)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.attention_mask = None
        self.position_bias = None
        self.layers = {} if layers is None else layers
        self.name = name
        self.built = False

    def build(
        self,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """用于完整构建Transformer模型
        layer_norm_*系列参数为实现Conditional Layer Normalization时使用，
        用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        # Call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        """模型的主要执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def apply(self, inputs, layer=None, arguments=None, **kwargs):
        """
        通过apply调用层会自动重用同名层
        :param inputs:    上一层的输出；
        :param layer:     要调用的层类名；
        :param arguments: 传递给layer.call的参数；
        :param kwargs:    传递给层初始化的参数。
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        arguments = arguments or {}
        name = kwargs.get('name')
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_mask(self, inputs=None):
        """定义每一层的Attention Mask
        """
        return self.attention_mask

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般用在定义相对位置编码层）
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """input和inputs属性设置
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """output和oututs属性设置
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_variable(self, checkpoint, name):
        """加载pre_model的checkpoint中单个变量
        """
        return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value):
        """用tensorflow中创建可用变量
        """
        return tf.Variable(value, name=name)

    def variable_mapping(self):
        """构建tf.keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            layer = self.layers[layer]
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]

            if isinstance(layer, MultiHeadAttention):
                """
                如果key_size不等于head_size，则可以通过
                正交矩阵将相应的权重投影到合适的shape。
                """
                count = 2
                if layer.use_bias:
                    count += 2
                heads = self.num_attention_heads
                head_size = self.attention_head_size
                key_size = self.attention_key_size
                W = np.linalg.qr(np.random.randn(key_size, head_size))[0].T
                if layer.attention_scale:
                    W = W * key_size**0.25 / head_size**0.25
                for i in range(count):
                    w, v = weights[i], values[i]
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if w_shape[-1] != v_shape[-1]:
                        pre_shape = w_shape[:-1]
                        v = v.reshape(pre_shape + (heads, head_size))
                        v = np.dot(v, W)
                        v = v.reshape(pre_shape + (heads * key_size,))
                        values[i] = v

            weight_value_pairs.extend(zip(weights, values))

        K.batch_set_value(weight_value_pairs)


class BERT(TransformerBlock):
    """通过transformer block构建BERT模型
    """
    def __init__(
        self,
        max_position,     # 序列最大长度
        with_pool=False,  # 是否包含Pool部分
        with_nsp=False,   # 是否包含NSP部分
        with_mlm=False,   # 是否包含MLM部分
        custom_position_ids=False,  # 是否自行传入位置id
        **kwargs          # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids

    def get_inputs(self):
        """
        token_ids和segment_ids为主要输入，
        （也允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = Input(shape=(self.sequence_length,), name='Input_Token')
        s_in = Input(shape=(self.sequence_length,), name='Input_Segment')

        if self.custom_position_ids:
            p_in = Input(shape=(self.sequence_length,), name='Input_Position')
            return [x_in, s_in, p_in]
        else:
            return [x_in, s_in]

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        x, s = inputs[:2]
        z = self.layer_norm_conds[0]
        if self.custom_position_ids:
            p = inputs[2]
        else:
            p = None

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding_Token'
        )
        s = self.apply(
            inputs=s,
            layer=Embedding,
            input_dim=2,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name='Embedding_Segment'
        )
        x = self.apply(inputs=[x, s], layer=Add, name='Embedding_Token_Segment')
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding_Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding_Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding_Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding_Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """
        BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """由剩余参数决定输出的形式
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool or self.with_nsp:
            # Pooler部分——提取CLS向量
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def load_variable(self, checkpoint, name):
        """加载pre_model的checkpoint中单个变量
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            if self.keep_tokens is None:
                return variable
            else:
                return variable[self.keep_tokens]
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        """用tensorflow中创建可用变量——根据输出格式定义
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value)

    def variable_mapping(self):
        """对官方bert的checkpoint中权重按格式进行映射
        """
        mapping = {
            'Embedding_Token': ['bert/embeddings/word_embeddings'],
            'Embedding_Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding_Position': ['bert/embeddings/position_embeddings'],
            'Embedding_Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding_Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


# 调用此函数直接构造xbert模型
def build_xbert_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
    ):
    """通过load bert's pre_trained checkpoint weights 和配置文件来直接构造bert模型
    :param config_path:        # bert参数文件路径
    :param checkpoint_path:    # bert预训练模型路径
    :param model:              # 构建模型的类型名字，默认加载google官方bert
    :param application:        # 默认使用encoder功能
    :param return_keras_model: # 默认返回tf.keras模型
    :param kwargs:
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')

    application = application.lower()

    models = {
        'bert': BERT,
    }

    if is_sting(model):
        model = model.lower()
        MODEL = models[model]
    else:
        MODEL = model

    transformer_block = MODEL(**configs)
    transformer_block.build(**configs)

    if checkpoint_path is not None:
        transformer_block.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer_block.model
    else:
        return transformer_block






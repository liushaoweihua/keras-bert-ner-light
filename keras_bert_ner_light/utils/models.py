# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: models.py.py
@Time: 2020/3/2 4:00 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



class NerBaseModel:
    """Bert Ner模型基础类
    """
    def __init__(self,
                 max_len,
                 embedding_dim,
                 numb_tags,
                 dropout_rate):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.numb_tags = numb_tags
        self.dropout_rate = dropout_rate

    def build(self):
        """Ner模型
        """
        x = Input(shape=(self.max_len, self.embedding_dim), name="Input-Features")
        x = Lambda(lambda X: X[:, 1:], name="Ignore-CLS")(x)
        x = self._task_layers(x)
        y = CRF(self.numb_tags, sparse_target=True, name="CRF")(x)
        model = Model(x, y)
        return model

    def _task_layers(self, layer):
        """下游网络层
        """
        raise NotImplementedError


class NerCnnModel(NerBaseModel):
    """Bert Ner模型 + Cnn下游模型
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 blocks,
                 *args,
                 **kwargs):
        super(NerCnnModel, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = blocks

    def _task_layers(self, layer):
        def dilation_conv1d(dilation_rate, name):
            return Conv1D(self.filters, self.kernel_size, padding="same", dilation_rate=dilation_rate, name=name)

        def idcnn_block(name):
            return [dilation_conv1d(1, name + "1"), dilation_conv1d(1, name + "2"), dilation_conv1d(2, name + "3")]

        stack_layers = []
        for layer_idx in range(self.blocks):
            name = "Idcnn-Block-%s-Layer-" % layer_idx
            idcnns = idcnn_block(name)
            cnn = idcnns[0](layer)
            cnn = idcnns[1](cnn)
            cnn = idcnns[2](cnn)
            stack_layers.append(cnn)
        stack_layers = concatenate(stack_layers, axis=-1)
        return stack_layers


class NerRnnModel(NerBaseModel):
    """Bert Ner模型 + Rnn下游模型
    """
    def __init__(self,
                 cell_type,
                 units,
                 num_hidden_layers,
                 *args,
                 **kwargs):
        super(NerRnnModel, self).__init__(*args, **kwargs)
        self.cell_type = cell_type.lower()
        allowed_cell_type = ["lstm", "gru"]
        assert self.cell_type in allowed_cell_type, "cell_type must be one of %s" % allowed_cell_type
        self.units = units
        self.num_hidden_layers = num_hidden_layers

    def _task_layers(self, layer):
        if self.cell_type == "lstm":
            cell = LSTM
            cell_name = "Lstm"
        elif self.cell_type == "gru":
            cell = GRU
            cell_name = "Gru"
        else:
            raise ValueError("cell_type should be 'lstm' or 'gru'.")
        rnn = layer
        for layer_idx in range(self.num_hidden_layers):
            name = cell_name + "-%s" % layer_idx
            rnn = Bidirectional(
                cell(units=self.units, return_sequences=True, recurrent_dropout=self.dropout_rate), name=name)(rnn)
        return rnn
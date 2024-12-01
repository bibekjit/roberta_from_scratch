import tensorflow as tf
from bert_base_layers import PositionalEmbeddingLayer
from encoder_layer import EncoderLayer
import numpy as np


class BERT(tf.keras.layers.Layer):

    def __init__(self,d_model,n_heads,vocab_size,input_len,
                 units,n_layers,**kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.units = units
        self.n_layers = n_layers
        self.input_len = input_len
        self.encodings = None

    def build(self,input_shape):
        super().build(input_shape)
        self.embedding = PositionalEmbeddingLayer(d_model=self.d_model,
                                                  vocab_size=self.vocab_size,
                                                  input_len=self.input_len)

        self.encoder_layers = [EncoderLayer(self.d_model,self.n_heads,self.units)
                               for _ in range(self.n_layers)]

    def call(self,x):

        x,mask = self.embedding(x)

        for i in range(self.n_layers):
            x = self.encoder_layers[i]([x,mask])

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'input_len': self.input_len,
            'vocab_size': self.vocab_size,
            'units': self.units,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers
        })
        return config

    def set_positional_encoding(self):

        l = self.input_len
        d = self.d_model

        mat = np.linspace(0, d - 1, d)
        mat = 1e-4 ** (2 * mat / d)
        mat = mat[np.newaxis, :]
        mat = np.tile(mat, [l, 1])

        for i in range(l):
            mat[i, :] = mat[i, :] * i

        for i in range(d):
            if i % 2 == 0:
                mat[:, i] = np.cos(mat[:, i])
            else:
                mat[:, i] = np.sin(mat[:, i])

        self.encodings = mat

        self.embedding.pos_emb_layer.set_weights([self.encodings])











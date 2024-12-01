import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout
import numpy as np


class PositionalEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self,d_model,vocab_size,input_len,**kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.input_len = input_len


    def build(self,input_shape):
        super().build(input_shape)
        self.pos_emb_layer = Embedding(self.input_len,self.d_model)
        self.word_emb_layer = Embedding(self.vocab_size,self.d_model)

    def call(self,x):
        mask = x != 0
        idx_pos = tf.range(0,self.input_len)
        x = self.word_emb_layer(x) + self.pos_emb_layer(idx_pos)
        return x,mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'vocab_size': self.vocab_size,
            'input_len': self.input_len
        })
        return config



class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self,d_model,n_heads,**kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.attention_scores = None

    def build(self,input_shape):
        super().build(input_shape)
        self.query = Dense(self.d_model)
        self.key = Dense(self.d_model)
        self.value = Dense(self.d_model)
        self.out = Dense(self.d_model)
        self.dropout = Dropout(0.1)

    def split_heads(self, inputs, batch_size):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.reshape(
            inputs, shape=(batch_size, inputs.shape[1], self.n_heads, self.d_head))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, x):

        x,mask = x

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        b = x.shape[0]

        # create attention heads
        q = self.split_heads(q,b)
        k = self.split_heads(k,b)
        v = self.split_heads(v,b)

        # apply attention mask and get softmax attention score
        score = tf.matmul(q,k,transpose_b=True)/self.d_model**0.5
        mask = tf.expand_dims(tf.expand_dims(mask,1),1)
        score = tf.where(mask,score,-1e9)
        score = self.dropout(score)
        score = tf.nn.softmax(score, axis=-1)
        self.attention_scores = score

        # get attention vector
        attn = tf.matmul(score,v)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn,(-1, tf.shape(attn)[1], self.d_model))
        attn = self.out(attn)
        return attn

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model':self.d_model,
            'n_heads':self.n_heads,
            'd_head':self.d_head,
            'attention_scores':self.attention_scores
        })
        return config


class FeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self,d_model,units,**kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.units = units

    def build(self,input_shape):
        super().build(input_shape)
        self.fc1 = Dense(self.units,activation='gelu')
        self.fc2 = Dense(self.d_model)
        self.dropout = Dropout(0.1)

    def call(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model':self.d_model,
            'units':self.units
        })
        return config










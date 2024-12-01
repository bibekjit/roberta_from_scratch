import tensorflow as tf
from bert_base_layers import FeedForwardLayer, MultiHeadSelfAttention


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,d_model,n_heads,units,**kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.units = units

    def build(self,input_shape):
        super().build(input_shape)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = MultiHeadSelfAttention(d_model=self.d_model,n_heads=self.n_heads)
        self.feedfwd = FeedForwardLayer(d_model=self.d_model, units=self.units)

    def call(self,inputs):
        x,mask = inputs
        attn = self.attention([x,mask])
        x = self.layer_norm_1(x + attn)
        ffwd = self.feedfwd(x)
        x = self.layer_norm_2(ffwd + x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'units': self.units
        })
        return config








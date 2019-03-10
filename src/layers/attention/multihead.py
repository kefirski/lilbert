import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers.experimental import LayerNormalization

from . import ScaledDotProductAttention


class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads, drop_p=0.1):
        """
        :param n_heads: Number of attention heads
        :param drop_p: dropout prob
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.drop_p = drop_p

    def build(self, input_shape):
        self.h_s = input_shape[-1]

        assert self.h_s % self.n_heads == 0

        self.p_s = int(self.h_s / self.n_heads)

        self.q = layers.Dense(self.h_s)
        self.k = layers.Dense(self.h_s)
        self.v = layers.Dense(self.h_s)

        self.attention = ScaledDotProductAttention()

        self.out = keras.Sequential(
            [layers.Dense(self.h_s), layers.Dropout(self.drop_p)]
        )

        self.layer_norm = LayerNormalization(-1)

    def call(self, q, k, v, residual=None, mask=None):
        """
        :param q: Tensor with shape of [batch_size, query_len, h_s]
        :param k: Tensor with shape of [batch_size, seq_len, h_s]
        :param v: Tensor with shape of [batch_size, seq_len, h_s]
        :param mask: Byte Tensor with shape of [batch_size, query_len, seq_len]
        :return: Tensor with shape of [batch_size, query_len, h_s]
        """

        b_s, q_len, _ = q.shape
        _, seq_len, _ = k.shape

        if residual is None:
            residual = q

        q = self.split_heads(self.q(q))
        k = self.split_heads(self.k(k))
        v = self.split_heads(self.v(v))

        if mask is not None:
            mask = tf.tile(tf.expand_dims(mask, 1), [1, self.n_heads, 1, 1])

        result = self.attention(q, k, v, mask)
        result = self.join_heads(result)
        result = self.out(result)

        return self.layer_norm(result + residual)

    def split_heads(self, input):
        b_s, s_l, h_s = input.shape
        return tf.transpose(
            tf.reshape(input, [b_s, s_l, self.n_heads, -1]), [0, 2, 1, 3]
        )

    def join_heads(self, input):
        b_s, _, s_l, _ = input.shape
        return tf.reshape(tf.transpose(input, [0, 2, 1, 3]), [b_s, s_l, -1])

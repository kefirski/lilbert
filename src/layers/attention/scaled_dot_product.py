from math import sqrt

import tensorflow as tf
import tensorflow.keras.layers as layers


class ScaledDotProductAttention(layers.Layer):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def build(self, input_shape):
        self.scaling = 1 / sqrt(input_shape[-1])

    def call(self, q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True) * self.scaling

        if mask is not None:
            paddings = tf.fill(mask.shape, -float("inf"))
            logits = tf.where(mask, paddings, logits)

        attention = tf.nn.softmax(logits, axis=-2)

        return tf.matmul(attention, v)

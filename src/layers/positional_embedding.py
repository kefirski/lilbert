import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class PostionalEmbedding(layers.Layer):
    def __init__(self, embedding_size):
        super(PostionalEmbedding, self).__init__()

        self.embedding_size = embedding_size
        self.test = [layers.Dense(10)]

    def call(self, input, t):
        b_s, s_l = input.shape

        input = tf.tile(tf.expand_dims(input, -1), [1, 1, self.embedding_size])

        js = tf.range(self.embedding_size, dtype=tf.float32)
        js = 10000 ** (2 * js / self.embedding_size)
        js = tf.tile(tf.reshape(js, [1, 1, -1]), [b_s, s_l, 1])

        _positional = input / js
        positional = np.zeros(_positional.shape)

        positional[:, :, 0::2] = tf.math.sin(_positional[:, :, 0::2])
        positional[:, :, 1::2] = tf.math.cos(_positional[:, :, 1::2])

        _additional = (t + 1) / js
        additional = np.zeros(_additional.shape)

        additional[:, :, 0::2] = tf.math.sin(additional[:, :, 0::2])
        additional[:, :, 1::2] = tf.math.cos(additional[:, :, 1::2])

        return tf.constant(positional) + tf.constant(additional)

    def pos_indices(self, input):
        b_s, s_l, *_ = input.shape
        return tf.tile(
            tf.expand_dims(tf.range(1, s_l + 1, dtype=tf.float32), 0), [b_s, 1]
        )

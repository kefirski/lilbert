import math

import tensorflow as tf
import tensorflow.keras.layers as layers


class GELU(layers.Layer):
    sqrt = math.sqrt(2 / math.pi)

    def call(self, x):
        return (
            0.5 * x * (1 + tf.math.tanh(self.sqrt * (x + 0.044715 * tf.math.pow(x, 3))))
        )

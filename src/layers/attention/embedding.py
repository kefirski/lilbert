import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class EmbeddingAttention(layers.layer):
    def __init__(self, n_lockups):
        super(EmbeddingAttention, self).__init__()

        self.n_lockups = n_lockups

    def build(self, input_shape):
        size = input_shape[-1]

        self.logits = keras.Sequential(
            layers.Dense(2 * size),
            layers.SELU(),
            layers.Dropout(0.4),
            layers.Dense(2 * size),
            layers.Activation("tanh"),
            layers.Dense(self.n_lockups),
        )

    def call(self, input, mask=None):
        b_s = input.shape[0]

        logits = tf.transpose(self.logits(input), [0, 2, 1])

        if mask is not None:
            mask = tf.tile(tf.expand_dim(mask, 1), [1, self.n_lockups, 1])
            paddings = tf.fill(mask.shape, -float("inf"))
            logits = tf.where(mask, paddings, logits)

        attention = tf.nn.softmax(logits, dim=-1)

        return tf.matmul(attention, input)

import tensorflow.keras.layers as layers

from .attention.multihead import MultiHeadAttention
from .position_wise import PositionWise


class EncoderLayer(layers.Layer):
    def __init__(self, n_heads, drop_p=0.1):
        super(EncoderLayer, self).__init__()

        self.drop_p = drop_p
        self.attention = MultiHeadAttention(n_heads, drop_p)

    def build(self, input_shape):
        h_s = input_shape[-1]
        self.position_wise = PositionWise(h_s * 4, self.drop_p)

    def call(self, input, pos, recur, mask=None):
        residual = input
        input = input + pos

        result = self.attention(input, input, input, residual=residual, mask=mask)
        return self.position_wise(result, recur), result

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .utils.gelu import GELU


class PositionWise(layers.Layer):
    def __init__(self, inner_size, drop_p=0.1):
        super(PositionWise, self).__init__()

        self.inner_size = inner_size
        self.drop_p = drop_p

    def build(self, input_shape):
        size = input_shape[-1]

        self.fc = keras.Sequential(
            layers.Dense(self.inner_size),
            GELU(),
            layers.Dense(size),
            layers.Dropout(self.drop_p),
        )

        self.layer_norm = LayerNormalization(-1)

    def call(self, input, residual=None):
        if residual is None:
            residual = input

        return self.layer_norm(self.fc(input) + residual)

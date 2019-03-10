import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from layers.encoder_layer import EncoderLayer
from layers.positional_embedding import PostionalEmbedding


class Model(layers.Layer):
    def __init__(self, n_l, n_heads, drop_p=0.1, **kwargs):
        super(Model, self).__init__()

        if "embedding_path" in kwargs:
            embedding = np.load(kwargs["embedding_path"])
            v_s, self.h_s = embedding.shape
            self.embedding = layers.Embedding(
                v_s, self.h_s, keras.initializers.Constant(embedding), trainable=False
            )
        else:
            v_s, self.h_s = kwargs["embedding_shape"]
            self.embedding = layers.Embedding(v_s, self.h_s)

        self.positional_embedding = PostionalEmbedding(self.h_s)

        self.layers = [EncoderLayer(n_heads, drop_p) for _ in range(n_l)]

        self.out = layers.Dense(v_s)

    def call(self, input, mask):

        input = self.embedding(input)
        pos = self.positional_embedding.pos_indices(input)

        mask = tf.tile(tf.expand_dims(mask, 1), [1, mask.shape[-1], 1])

        recur = tf.zeros_like(input)
        for i, layer in enumerate(self.layers):
            input, recur = layer(input, self.positional_embedding(pos, i), recur, mask)

        return self.out(input)

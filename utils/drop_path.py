import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if self.drop_prob == 0.0 or not training:
            return inputs
        else:
            batch_size = tf.shape(inputs)[0]
            keep_prob = 1 - self.drop_prob
            path_mask_shape = (batch_size,) + (1,) * (len(tf.shape(inputs)) - 1)
            path_mask = tf.floor(backend.random_bernoulli(path_mask_shape, p=keep_prob))
            outputs = (
                tf.math.divide(tf.cast(inputs, dtype=tf.float32), keep_prob) * path_mask
            )
            return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_prob": self.drop_prob,
            }
        )
        return config

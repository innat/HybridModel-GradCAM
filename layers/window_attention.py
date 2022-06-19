import tensorflow as tf
from tensorflow.keras import layers


class WindowAttention(layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        return_attention_scores=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.return_attention_scores = return_attention_scores
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )

        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1]
        )
        super().build(input_shape)

    def get_relative_position_index(self, window_height, window_width):
        x_x, y_y = tf.meshgrid(range(window_height), range(window_width))
        coords = tf.stack([y_y, x_x], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])

        x_x = (relative_coords[:, :, 0] + window_height - 1) * (2 * window_width - 1)
        y_y = relative_coords[:, :, 1] + window_width - 1
        relative_coords = tf.stack([x_x, y_y], axis=-1)

        return tf.reduce_sum(relative_coords, axis=-1)

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)

        if self.return_attention_scores:
            return x_qkv, attn
        else:
            return x_qkv

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "scale": self.scale,
            }
        )
        return config

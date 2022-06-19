import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers.swin_blocks import SwinTransformer
from utils.model_utils import *
from utils.patch import PatchEmbedding
from utils.patch import PatchExtract
from utils.patch import PatchMerging


class HybridSwinTransformer(keras.Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(name=model_name, **kwargs)
        # base models
        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=keras.Input((params.image_size, params.image_size, 3)),
        )

        # base model with compatible output which will be an input of transformer model
        self.new_base = keras.Model(
            [base.inputs],
            [base.get_layer("block6a_expand_activation").output, base.output],
            name="efficientnet",
        )

        # stuff of swin transformers
        self.patch_extract = PatchExtract(patch_size)
        self.patch_embedds = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)
        self.patch_merging = PatchMerging(
            (num_patch_x, num_patch_y), embed_dim=embed_dim
        )

        # swin blocks containers
        self.swin_sequences = keras.Sequential(name="swin_blocks")
        for i in range(shift_size):
            self.swin_sequences.add(
                SwinTransformer(
                    dim=embed_dim,
                    num_patch=(num_patch_x, num_patch_y),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=i,
                    num_mlp=num_mlp,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                )
            )

        # swin block's head
        self.swin_head = keras.Sequential(
            [
                layers.GlobalAveragePooling1D(),
                layers.AlphaDropout(0.5),
                layers.BatchNormalization(),
            ],
            name="swin_head",
        )

        # base model's (cnn model) head
        self.conv_head = keras.Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.AlphaDropout(0.5),
            ],
            name="conv_head",
        )

        # classifier
        self.classifier = layers.Dense(
            params.class_number, activation=None, dtype="float32"
        )
        self.build_graph()

    def call(self, inputs, training=None, **kwargs):
        x, base_gcam_top = self.new_base(inputs)
        x = self.patch_extract(x)
        x = self.patch_embedds(x)
        x = self.swin_sequences(tf.cast(x, dtype=tf.float32))
        x, swin_gcam_top = self.patch_merging(x)

        swin_top = self.swin_head(x)
        conv_top = self.conv_head(base_gcam_top)
        preds = self.classifier(tf.concat([swin_top, conv_top], axis=-1))

        if training:  # training phase
            return preds
        else:  # inference phase
            return preds, base_gcam_top, swin_gcam_top

    def build_graph(self):
        x = keras.Input(shape=(params.image_size, params.image_size, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))


class GradientAccumulation(HybridSwinTransformer):
    """ref: https://gist.github.com/innat/ba6740293e7b7b227829790686f2119c"""

    def __init__(self, n_gradients, **kwargs):
        super().__init__(**kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.trainable_variables
        ]

    def train_step(self, data):
        # track accumulation step update
        self.n_acum_step.assign_add(1)

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to -
        # update the variables otherwise do nothing
        tf.cond(
            tf.equal(self.n_acum_step, self.n_gradients),
            self.apply_accu_gradients,
            lambda: None,
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # Update weights
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset accumulation step
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i], dtype=tf.float32)
            )

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_pred, base_gcam_top, swin_gcam_top = self(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

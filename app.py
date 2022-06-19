import os

import gdown
import gradio as gr
import tensorflow as tf

from config import Parameters
from models.hybrid_model import GradientAccumulation
from utils.model_utils import *
from utils.viz_utils import make_gradcam_heatmap
from utils.viz_utils import save_and_display_gradcam

image_size = Parameters().image_size
str_labels = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
]


def get_model():
    """Get the model."""
    model = GradientAccumulation(
        n_gradients=params.num_grad_accumulation, model_name="HybridModel"
    )
    _ = model(tf.ones((1, params.image_size, params.image_size, 3)))[0].shape
    return model


def get_model_weight(model_id):
    """Get the trained weights."""
    if not os.path.exists("model.h5"):
        model_weight = gdown.download(id=model_id, quiet=False)
    else:
        model_weight = "model.h5"
    return model_weight


def load_model(model_id):
    """Load trained model."""
    weight = get_model_weight(model_id)
    model = get_model()
    model.load_weights(weight)
    return model


def image_process(image):
    """Image preprocess for model input."""
    image = tf.cast(image, dtype=tf.float32)
    original_shape = image.shape
    image = tf.image.resize(image, [image_size, image_size])
    image = image[tf.newaxis, ...]
    return image, original_shape


def predict_fn(image):
    """A predict function that will be invoked by gradio."""
    loaded_model = load_model(model_id="1y6tseN0194T6d-4iIh5wo7RL9ttQERe0")
    loaded_image, original_shape = image_process(image)

    heatmap_a, heatmap_b, preds = make_gradcam_heatmap(loaded_image, loaded_model)
    int_label = tf.argmax(preds, axis=-1).numpy()[0]
    str_label = str_labels[int_label]

    overaly_a = save_and_display_gradcam(
        loaded_image[0], heatmap_a, image_shape=original_shape[:2]
    )
    overlay_b = save_and_display_gradcam(
        loaded_image[0], heatmap_b, image_shape=original_shape[:2]
    )

    return [f"Predicted: {str_label}", overaly_a, overlay_b]


iface = gr.Interface(
    fn=predict_fn,
    inputs=gr.inputs.Image(label="Input Image"),
    outputs=[
        gr.outputs.Label(label="Prediction"),
        gr.inputs.Image(label="CNN GradCAM"),
        gr.inputs.Image(label="Transformer GradCAM"),
    ],
    title="Hybrid EfficientNet Swin Transformer Demo",
    description="The model is trained on tf_flowers dataset.",
    examples=[
        ["examples/dandelion.jpg"],
        ["examples/sunflower.jpg"],
        ["examples/tulip.jpg"],
        ["examples/daisy.jpg"],
        ["examples/rose.jpg"],
    ],
)
iface.launch()

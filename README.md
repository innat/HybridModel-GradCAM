
# Hybrid EfficientNet Swin Transformer (HENetSwinT)

![](https://user-images.githubusercontent.com/17668390/149625554-b9c7074a-2137-49d5-8726-a3fbfa3f9a4c.gif)
<div align="center">
  Figure: Grad-CAMs. Left (Input), Middle (CNN), Right (CNN + Swin Transformer).
</div>

# Code

This repo provides and unofficial implementation of Hybrid EfficientNet-Swin-Transformer in `TensorFLow 2`. Reference paper: https://arxiv.org/pdf/2110.03786.pdf.

1. [TF.Keras: Hybrid EfficientNet Swin Transformer TPU](https://www.kaggle.com/ipythonx/tf-keras-hybrid-efficientnet-swin-transformer-tpu)
2. [TF: Hybrid EfficientNet Swin-Transformer : GradCAM](https://www.kaggle.com/ipythonx/tf-hybrid-efficientnet-swin-transformer-gradcam)


## DISCLAIMER
- The implementation of swin-transformer in tf.keras is mostly borrowed from [VcampSoldiers/Swin-Transformer-Tensorflow](https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow) and [keras-code-examples](https://keras.io/examples/vision/swin_transformers/).
- Please note, none of the un-official implementation is efficient enough; there can be some possible bugs. So, consider it if you use this. Please refer to the official implementation in pytorch in that case.

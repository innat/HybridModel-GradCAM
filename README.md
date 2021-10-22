# What is this notebook about?¶
- It's a code example script (demonstration purpose don't belon to any comp.).
- A minimal implementation of Hybrid Efficientnet Swin-Transformer in tf.keras
- Grad-CAM of the Swin-Transformer.
- Use CutMix, MixUp augmentaiton. How does swin react on them!
- Check out [this thread](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277917) for more details. Below is the overall diagram. And in this notebook, we're trying to implement it quickly as possible. Please note, the implementation would be not optimal but hopefully it'd ok to get started on it.

![](https://i.imgur.com/2iXNuBA.png)

## DISCLAIMER
- The implementation of swin-transformer in tf.keras is mostly borrowed from [VcampSoldiers/Swin-Transformer-Tensorflow](https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow) and [keras-code-examples](https://keras.io/examples/vision/swin_transformers/).
- Please note, none of the un-official implementation is efficient enough; there are some possible bugs. So, consider it if you use this. Please refer to the official implementation in pytorch in that case.
- Also, this is just for demonstration purpose.

## Runing Code 

- [Kaggle](https://www.kaggle.com/ipythonx/tf-hybrid-swintransformer-cutmix-mixup-gradcam).
- [Colab]()


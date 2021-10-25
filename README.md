# EfficientNet Hybrid-Swin-Transformers Implementation in TensorFlow 2 (Keras).

This repo provides and unofficial implementation of Hybrid-Swin-Transformer in `TensorFLow 2`. Reference paper: https://arxiv.org/pdf/2110.03786.pdf

**What is this notebook about?**

- It's a code example script (demonstration purpose).
- A minimal implementation of **Hybrid Efficientnet Swin-Transformer** in `tf.keras`
- Inspecting via Grad-CAM of the Swin-Transformer.
- Use **CutMix**, **MixUp** augmentaiton. How does swin react on them!
- Check out [this thread](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277917) for more details. 

## Model EfficientNet Hybrid-Swin-Transformers

**In paper**: 

![](https://i.imgur.com/2iXNuBA.png)


**In ours**:

With `EfficientNet B0` (input: **512**), middle layer: `'block2a_expand_activation'` (output: **256**)  and further use it for swin transformer. 

![model](https://user-images.githubusercontent.com/17668390/138736618-0a72cba4-84ab-42b2-8728-b981c9392ea1.png)

## Runing Code 
- Run: `python efficientnet_hybrid_swin.py`. Or, 
- Dataset: [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score)
  - [Kaggle Notebook](https://www.kaggle.com/ipythonx/tf-hybrid-swintransformer-cutmix-mixup-gradcam).
  - [Colab Notebook](https://colab.research.google.com/drive/1Q_V5FcEtiflitPtc_utd0zrY8JFGahLv?usp=sharing)

## Grad-CAM of Hybrid-Swin 
![download](https://user-images.githubusercontent.com/17668390/138469478-37180400-54ed-4cf4-9cf5-2eb562f10ab8.png)



## DISCLAIMER
- The implementation of swin-transformer in tf.keras is mostly borrowed from [VcampSoldiers/Swin-Transformer-Tensorflow](https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow) and [keras-code-examples](https://keras.io/examples/vision/swin_transformers/).
- Please note, none of the un-official implementation is efficient enough; there are some possible bugs. So, consider it if you use this. Please refer to the official implementation in pytorch in that case.

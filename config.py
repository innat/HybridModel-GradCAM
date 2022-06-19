import numpy as np
import tensorflow as tf


class Parameters:
    # data level
    image_count = 3670
    image_size = 384
    batch_size = 12
    num_grad_accumulation = 8
    label_smooth = 0.05
    class_number = 5
    val_split = 0.2
    autotune = tf.data.AUTOTUNE

    # hparams
    epochs = 10
    lr_sched = "cosine_restart"
    lr_base = 0.016
    lr_min = 0
    lr_decay_epoch = 2.4
    lr_warmup_epoch = 5
    lr_decay_factor = 0.97

    scaled_lr = lr_base * (batch_size / 256.0)
    scaled_lr_min = lr_min * (batch_size / 256.0)
    num_validation_sample = int(image_count * val_split)
    num_training_sample = image_count - num_validation_sample
    train_step = int(np.ceil(num_training_sample / float(batch_size)))
    total_steps = train_step * epochs

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:37:28 2021

@author: innat
"""
from tensorflow.keras import Model, Input, applications, layers 
from swin_blocks import *


img_size = 512
class_number = 10

class HybridSwinTransformer(Model):
    def __init__(self):
        super(HybridSwinTransformer, self).__init__()
        # base models 
        self.inputx = Input((img_size, img_size, 3), name='input_hybrids')
        base = applications.EfficientNetB0(include_top=False,
                                                weights=None,
                                                input_tensor=self.inputx
                                               )
        # base model with compatible output which will be an input of transformer model 
        self.new_base = Model(
            [base.inputs], 
            [base.get_layer('block2a_expand_activation').output], # output with 256 feat_maps
            name='efficientnet'
        )

        # stuff of swin transformers 
        self.patch_extract  = PatchExtract(patch_size)
        self.patch_embedds  = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)
        self.patch_merging  = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)
        self.swin_sequences = Sequential(
            [
                SwinTransformer(
                    dim=embed_dim,
                    num_patch=(num_patch_x, num_patch_y),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=i,
                    num_mlp=num_mlp,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate) for i in range(shift_size)
            ], name='swin_blocks')
        
        self.tail = Sequential(
            [
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.2),
                layers.BatchNormalization(),
                layers.Dense(class_number)
            ], name='head'
        )
        
    # ma-ma calling 
    def call(self, inputs, training=None, **kwargs):
        x = self.new_base(inputs)
        x = self.patch_extract(x)
        x = self.patch_embedds(x)
        x = self.swin_sequences(x)
        x = self.patch_merging(x)
        
        if training: # training phase 
            return self.tail(x)
        else: # inference phase, 
            return self.tail(x), x

    def build_graph(self):
        x = Input(shape=(img_size, img_size, 3))
        return Model(inputs=[x], outputs=self.call(x))
    
tf.keras.backend.clear_session()
model = HybridSwinTransformer()
print(model(tf.ones((2, img_size, img_size, 3)))[0].shape)
display(tf.keras.utils.plot_model(model.build_graph(), 
                                  show_shapes=True, 
                                  show_layer_names=True,  expand_nested=True))
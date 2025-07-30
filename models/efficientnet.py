import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, Reshape, \
                        Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Average
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, softmax

import math
IMG_SIZE=(224,224,3)

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = K.int_shape(input_feature)[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    # Use Lambda layers to wrap tf functions
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    # Concatenate along channel axis
    concat = Concatenate(axis=-1)([avg_pool, max_pool])  # (H, W, 2)

    # Apply 7x7 convolution → 1-channel spatial attention map
    attention = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)  # (H, W, 1)

    # Multiply attention map with input feature map
    output = layers.Multiply()([input_feature, attention])

    return output
    
    
class ECALayer(tf.keras.layers.Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = input_shape[-1]
        t = int(abs((math.log2(channels) / self.gamma) + self.b))
        self.k_size = t if t % 2 == 1 else t + 1
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.k_size,
            padding='same',
            use_bias=False
        )

    def call(self, x):
        # x: [B, H, W, C]
        squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # [B, 1, 1, C]
        squeeze = tf.transpose(squeeze, [0, 3, 1, 2])            # [B, C, 1, 1]
        squeeze = tf.squeeze(squeeze, axis=[2, 3])               # [B, C]
        squeeze = tf.expand_dims(squeeze, axis=-1)              # [B, C, 1]
        attn = self.conv1d(squeeze)                              # [B, C, 1]
        attn = tf.nn.sigmoid(attn)                               # [B, C, 1]
        attn = tf.reshape(attn, [-1, 1, 1, x.shape[-1]])         # [B, 1, 1, C]
        return x * attn



def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature= ECALayer()(cbam_feature)
    cbam_feature = spatial_attention(cbam_feature)

    return cbam_feature

from tensorflow.keras.applications import efficientnet_v2

def EfficientNetV2Base():
    efficientnet = efficientnet_v2.EfficientNetV2B0(input_shape=IMG_SIZE, weights=None, include_top=False, include_preprocessing=False)
    return efficientnet

def buildModel():
    # Load the Inception model with weights pre-trained on ImageNet.
    base_model = EfficientNetV2Base() 
    base_model.trainable = False
    print("Base Model:", base_model.name)

    # Define the input layer.
    inputs = keras.Input(shape=IMG_SIZE)
    x = inputs
    
    # x = layers.Resizing(IMG_SIZE[0], IMG_SIZE[1], name='PP_Resize')(x)
    # x = layers.Rescaling(1./255, name='PP_Rescale_down')(x)

    # Pass the input through the pre-trained InceptionV3 model.
    x = base_model(x, training=False) 
    ## There is already dropout in end of base_model

    original_features = x
    eca=ECALayer()(x)
    cbam = spatial_attention(x)
    result1 = Add()([eca, cbam])
    result2 = cbam_block(x, ratio=8)
    cbam_feature = Add()([result1, result2])
    x = Concatenate()([cbam_feature, original_features])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, name='dropout')(x) 
    
    outputs = layers.Dense(9, activation='softmax')(x)

    # Create the model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(loss='categoricalcrossentropy', optimizer='adam', metrics=['accuracy'])
    return model
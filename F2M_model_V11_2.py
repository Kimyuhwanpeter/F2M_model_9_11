# -*- coding:utf-8 -*-
import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def attention_residual_block(input, dilation=1, filters=256):

    h = input

    h_attenion_layer = tf.reduce_mean(input, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, padding="same", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    
    return (h*h_attenion_layer) + input

def decode_residual_block(input, dilation=1, filters=256):

    h = input

    h_attenion_layer = tf.reduce_mean(input, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, padding="same", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    
    return (h*h_attenion_layer) + input

def F2M_generator_V2(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    for i in range(6):
        h = attention_residual_block(h, dilation=(i+1)*2, filters=256)

    # adative contrast 를 적용해볼까?


    h_mid = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]

    for i in range(2):
        h = decode_residual_block(h, dilation=(i+1)*4, filters=128)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False,
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="valid")(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=[h, h_mid])

def F2M_discriminator(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = F2M_generator_V2()
model.summary()
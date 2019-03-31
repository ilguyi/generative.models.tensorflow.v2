from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers



class Conv(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides, padding='same',
               activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5):
    super(Conv, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    assert activation in ['relu', 'leaky_relu', 'none']
    self.activation = activation

    self.conv = layers.Conv2D(filters=filters,
                              kernel_size=(kernel_size, kernel_size),
                              strides=strides,
                              padding=padding,
                              kernel_initializer=tf.random_normal_initializer(0., 0.02),
                              use_bias=not self.apply_batchnorm)
    if self.apply_batchnorm:
      self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                 epsilon=norm_epsilon)

  def call(self, x, training=True):
    # convolution
    x = self.conv(x)

    # batchnorm
    if self.apply_batchnorm:
      x = self.batchnorm(x, training=training)

    # activation
    if self.activation == 'relu':
      x = tf.nn.relu(x)
    elif self.activation == 'leaky_relu':
      x = tf.nn.leaky_relu(x)
    else:
      pass

    return x



class ConvTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides=2, padding='same',
               activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5):
    super(ConvTranspose, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    assert activation in ['relu', 'sigmoid', 'none']
    self.activation = activation
    self.up_conv = layers.Conv2DTranspose(filters=filters,
                                          kernel_size=(kernel_size, kernel_size),
                                          strides=strides,
                                          padding=padding,
                                          kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                          use_bias=not self.apply_batchnorm)
    if self.apply_batchnorm:
      self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                 epsilon=norm_epsilon)

  def call(self, x, training=True):
    # conv transpose
    x = self.up_conv(x)
    
    # batchnorm
    if self.apply_batchnorm:
      x = self.batchnorm(x, training=training)
      
    # activation
    if self.activation == 'relu':
      x = tf.nn.relu(x)
    elif self.activation == 'sigmoid':
      x = tf.nn.sigmoid(x)
    else:
      pass
    
    return x



class Dense(tf.keras.Model):
  def __init__(self, units, activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5):
    super(Dense, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    assert activation in ['relu', 'leaky_relu', 'none']
    self.activation = activation

    self.dense = layers.Dense(units=units,
                              kernel_initializer=tf.random_normal_initializer(0., 0.02),
                              use_bias=not self.apply_batchnorm)
    if self.apply_batchnorm:
      self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                 epsilon=norm_epsilon)

  def call(self, x, training=True):
    # dense
    x = self.dense(x)

    # batchnorm
    if self.apply_batchnorm:
      x = self.batchnorm(x, training=training)

    # activation
    if self.activation == 'relu':
      x = tf.nn.relu(x)
    elif self.activation == 'leaky_relu':
      x = tf.nn.leaky_relu(x)
    else:
      pass

    return x




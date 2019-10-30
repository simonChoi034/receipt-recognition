import tensorflow as tf
from tensorflow.keras.layers import Input
from .layers import conv2d, residual_block

def create_model(input):
    inputs = Input(shape=[None, None, 3])
    x = inputs
    # 2 convolution layers
    x = conv2d(x, filters=32, kernel_size=3)
    x = conv2d(x, filters=64, kernel_size=3, strides=2)

    # 1st residual block cluster
    x = residual_block(x, filters=[32, 64], kernel_size=[1, 3])
    x = conv2d(x, filters=128, kernel_size=3, strides=2)

    # 2nd residual block cluster
    for i in range(0, 2):
        x = residual_block(x, filters=[64, 128], kernel_size=[1, 3])
    x = conv2d(x, filters=256, kernel_size=3, strides=2)

    # 3rd residual block cluster
    for i in range(0, 8):
        x = residual_block(x, filters=[128, 256], kernel_size=[1, 3])
    # Scale 3 output for anchor prediction
    scale3 = x
    x = conv2d(x, filters=512, kernel_size=3, strides=2)

    # 4th residual block cluster
    for i in range(0, 8):
        x = residual_block(x, filters=[256, 512], kernel_size=[1, 3])
    # Scale 2 output for anchor prediction
    scale2 = x
    x = conv2d(x, filters=1024, kernel_size=3, strides=2)

    # 5th residual block cluster
    for i in range(0, 4):
        x = residual_block(x, filters=[512, 1024], kernel_size=[1, 3])
    # Scale 1 output for anchor prediction
    scale1 = x

    return tf.keras.Model(inputs=inputs, outputs=[scale3, scale2, scale1])
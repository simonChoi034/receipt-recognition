from tensorflow.keras import layers

from .layers import MyConv2D, ResidualBlock


class Darknet53(layers.Layer):
    def __init__(self, name='darknet-53', **kwargs):
        super(Darknet53, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        # 2 convolution layers
        x = MyConv2D(filters=32, kernel_size=3)(inputs)
        x = MyConv2D(filters=64, kernel_size=3, strides=2)(x)

        # 1st residual block cluster
        x = ResidualBlock(filters=[32, 64], kernel_size=[1, 3])(x)
        x = MyConv2D(filters=128, kernel_size=3, strides=2)(x)

        # 2nd residual block cluster
        for i in range(0, 2):
            x = ResidualBlock(filters=[64, 128], kernel_size=[1, 3])(x)
        x = MyConv2D(filters=256, kernel_size=3, strides=2)(x)

        # 3rd residual block cluster
        for i in range(0, 8):
            x = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])(x)
        # Scale 3 output for anchor prediction
        scale3 = x
        x = MyConv2D(filters=512, kernel_size=3, strides=2)(x)

        # 4th residual block cluster
        for i in range(0, 8):
            x = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])(x)
        # Scale 2 output for anchor prediction
        scale2 = x
        x = MyConv2D(filters=1024, kernel_size=3, strides=2)(x)

        # 5th residual block cluster
        for i in range(0, 4):
            x = ResidualBlock(filters=[512, 1024], kernel_size=[1, 3])(x)
        # Scale 1 output for anchor prediction
        scale1 = x

        return scale3, scale2, scale1

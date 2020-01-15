from tensorflow.keras import layers

from .layers import MyConv2D, ResidualBlock


class Darknet53(layers.Layer):
    def __init__(self, name='darknet-53', **kwargs):
        super(Darknet53, self).__init__(name=name, **kwargs)
        self.conv_1 = MyConv2D(filters=32, kernel_size=3)
        self.conv_2 = MyConv2D(filters=64, kernel_size=3, strides=2)
        self.conv_3 = MyConv2D(filters=128, kernel_size=3, strides=2)
        self.conv_4 = MyConv2D(filters=256, kernel_size=3, strides=2)
        self.conv_5 = MyConv2D(filters=512, kernel_size=3, strides=2)
        self.conv_6 = MyConv2D(filters=1024, kernel_size=3, strides=2)
        self.res_1 = ResidualBlock(filters=[32, 64], kernel_size=[1, 3])
        self.res_2 = [ResidualBlock(filters=[64, 128], kernel_size=[1, 3]) for _ in range(0, 2)]
        self.res_3 = [ResidualBlock(filters=[128, 256], kernel_size=[1, 3]) for _ in range(0, 8)]
        self.res_4 = [ResidualBlock(filters=[256, 512], kernel_size=[1, 3]) for _ in range(0, 8)]
        self.res_5 = [ResidualBlock(filters=[512, 1024], kernel_size=[1, 3]) for _ in range(0, 4)]

    def call(self, inputs, training=False, **kwargs):
        # 2 convolution layers
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)

        # 1st residual block cluster
        x = self.res_1(x, training=training)
        x = self.conv_3(x, training=training)

        # 2nd residual block cluster
        for res in self.res_2:
            x = res(x, training=training)
        x = self.conv_4(x, training=training)

        # 3rd residual block cluster
        for res in self.res_3:
            x = res(x, training=training)
        # Scale 3 output for anchor prediction
        scale3 = x
        x = self.conv_5(x, training=training)

        # 4th residual block cluster
        for res in self.res_4:
            x = res(x, training=training)
        # Scale 2 output for anchor prediction
        scale2 = x
        x = self.conv_6(x, training=training)

        # 5th residual block cluster
        for res in self.res_5:
            x = res(x, training=training)
        # Scale 1 output for anchor prediction
        scale1 = x

        return scale1, scale2, scale3

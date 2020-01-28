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
        # 1
        self.res_1 = ResidualBlock(filters=[32, 64], kernel_size=[1, 3])
        # 2
        self.res_2 = ResidualBlock(filters=[64, 128], kernel_size=[1, 3])
        self.res_3 = ResidualBlock(filters=[64, 128], kernel_size=[1, 3])
        # 8
        self.res_4 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_5 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_6 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_7 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_8 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_9 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_10 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        self.res_11 = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])
        # 8
        self.res_12 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_13 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_14 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_15 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_16 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_17 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_18 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        self.res_19 = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])
        # 4
        self.res_20 = ResidualBlock(filters=[512, 1024], kernel_size=[1, 3])
        self.res_21 = ResidualBlock(filters=[512, 1024], kernel_size=[1, 3])
        self.res_22 = ResidualBlock(filters=[512, 1024], kernel_size=[1, 3])
        self.res_23 = ResidualBlock(filters=[512, 1024], kernel_size=[1, 3])

    def call(self, inputs, training=False, **kwargs):
        # 2 convolution layers
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)

        # 1st residual block cluster
        x = self.res_1(x, training=training)
        x = self.conv_3(x, training=training)

        # 2nd residual block cluster
        x = self.res_2(x, training=training)
        x = self.res_3(x, training=training)
        x = self.conv_4(x, training=training)

        # 3rd residual block cluster
        x = self.res_4(x, training=training)
        x = self.res_5(x, training=training)
        x = self.res_6(x, training=training)
        x = self.res_7(x, training=training)
        x = self.res_8(x, training=training)
        x = self.res_9(x, training=training)
        x = self.res_10(x, training=training)
        x = self.res_11(x, training=training)
        # Scale 3 output for anchor prediction
        output_large = x
        x = self.conv_5(x, training=training)

        # 4th residual block cluster
        x = self.res_12(x, training=training)
        x = self.res_13(x, training=training)
        x = self.res_14(x, training=training)
        x = self.res_15(x, training=training)
        x = self.res_16(x, training=training)
        x = self.res_17(x, training=training)
        x = self.res_18(x, training=training)
        x = self.res_19(x, training=training)
        # Scale 2 output for anchor prediction
        output_medium = x
        x = self.conv_6(x, training=training)

        # 5th residual block cluster
        x = self.res_20(x, training=training)
        x = self.res_21(x, training=training)
        x = self.res_22(x, training=training)
        x = self.res_23(x, training=training)
        # Scale 1 output for anchor prediction
        output_small = x

        return output_small, output_medium, output_large

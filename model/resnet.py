from tensorflow.keras import layers

from .layers import MyConv2D, ResidualBlock


class Resnet18(layers.Layer):
    def __init__(self, name='resnet-18', **kwargs):
        super(Resnet18, self).__init__(name=name, **kwargs)
        self.conv1 = MyConv2D(filters=64, kernel_size=3)
        self.res1 = ResidualBlock(filters=64, kernel_size=3)
        self.conv2 = MyConv2D(filters=128, kernel_size=3, strides=2)
        self.res2 = ResidualBlock(filters=128, kernel_size=3)
        self.res3 = ResidualBlock(filters=128, kernel_size=3)
        self.conv3 = MyConv2D(filters=256, kernel_size=3, strides=2)
        self.res4 = ResidualBlock(filters=256, kernel_size=3)
        self.res5 = ResidualBlock(filters=256, kernel_size=3)
        self.conv4 = MyConv2D(filters=512, kernel_size=3, strides=2)
        self.res6 = ResidualBlock(filters=512, kernel_size=3)
        self.res7 = ResidualBlock(filters=512, kernel_size=3)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.res1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.conv3(x, training=training)
        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.conv4(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)

        return x

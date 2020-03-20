import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2


class MyConv2D(layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            padding='same',
            activation=True,
            apply_batchnorm=True):
        super(MyConv2D, self).__init__()
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm
        self.conv2d = Conv2D(
            filters,
            kernel_size,
            strides,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=tf.random_normal_initializer(0., 0.05),
            kernel_regularizer=l2(0.0005)
        )
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv2d(inputs)
        if self.apply_batchnorm:
            x = self.batch_norm(x, training=training)

        if self.activation:
            x = self.leaky_relu(x)

        return x


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self.filters = [filters, filters] if isinstance(filters, int) else filters
        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.conv1 = MyConv2D(filters=self.filters[0], kernel_size=self.kernel_size[0])
        self.conv2 = MyConv2D(filters=self.filters[1], kernel_size=self.kernel_size[1], activation=False)
        self.shortcut = MyConv2D(filters=self.filters[1], kernel_size=1)
        self.leaky_relu = LeakyReLU()

    def call(self, inputs, training=False, **kwargs):
        shortcut = self.shortcut(inputs)

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        # residual shortcut
        x += shortcut
        x = self.leaky_relu(x)

        return x

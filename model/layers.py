import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, Concatenate
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
            apply_batchnorm=True,
            name='conv2d',
            **kwargs):
        super(MyConv2D, self).__init__(name=name, **kwargs)
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm
        self.conv2d = Conv2D(
            filters,
            kernel_size,
            strides,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=tf.random_normal_initializer(0., 0.05),
            kernel_regularizer=l2()
        )
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        if self.apply_batchnorm:
            x = self.batch_norm(x, training=training)

        if self.activation:
            x = self.leaky_relu(x)

        return x


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, name='residual-block', **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
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


class CBAM(layers.Layer):
    def __init__(self, filters, reduction, name='convolutional-block-attention-module', **kwargs):
        super(CBAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.reduction = reduction
        self.channel_conv1 = MyConv2D(filters=filters // reduction, kernel_size=1, apply_batchnorm=False)
        self.channel_conv2 = MyConv2D(filters=filters, kernel_size=1, apply_batchnorm=False, activation=False)
        self.spatial_conv = MyConv2D(filters=1, kernel_size=7, apply_batchnorm=False, activation=False)
        self.sigmoid = Activation(tf.nn.sigmoid)
        self.concat = Concatenate()

    def call(self, inputs, training=False, **kwargs):
        # channel attention
        x_mean = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
        x_mean = self.channel_conv1(x_mean, training=training)
        x_mean = self.channel_conv2(x_mean, training=training)

        x_max = tf.reduce_max(inputs, axis=(1, 2), keepdims=True)
        x_max = self.channel_conv1(x_max, training=training)
        x_max = self.channel_conv2(x_max, training=training)

        x = x_mean + x_max
        x = self.sigmoid(x)
        x = tf.multiply(inputs, x)

        # spatial attention
        y_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        y_max = tf.reduce_max(x, axis=-1, keepdims=True)
        y = self.concat([y_mean, y_max])
        y = self.spatial_conv(y, training=training)
        y = self.sigmoid(y)
        y = tf.multiply(x, y)

        return y

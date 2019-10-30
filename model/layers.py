import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

def conv2d(x, filters, kernel_size, strides=1, activation=True, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    block = Sequential()
    block.add(
        Conv2D(
            filters,
            kernel_size,
            strides,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    if apply_batchnorm:
        block.add(BatchNormalization())

    if activation:
        block.add(LeakyReLU())

    output = block(x)

    return output

def residual_block(x, filters, kernel_size):
    if isinstance(filters, int):
        filters = [filters, filters]

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    shortcut = x

    x = conv2d(x, filters[0], kernel_size[0])
    x = conv2d(x, filters[1], kernel_size[1], activation=False)
    # residual shortcut
    x += shortcut
    x = LeakyReLU()(x)

    return x
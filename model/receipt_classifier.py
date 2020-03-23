import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, GRU, Embedding, TimeDistributed, RepeatVector, \
    Layer, ZeroPadding2D, Dropout, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.regularizers import l2

from model.crf import CRF
from model.layers import MyConv2D


class GridClassifier(tf.keras.Model):
    def __init__(self, num_class, gird_size, name='cnn-classifier', **kwargs):
        super(GridClassifier, self).__init__(name=name, **kwargs)
        self.conv_block = [MyConv2D(filters=256, kernel_size=[3, 5]) for _ in range(4)]
        self.dilated_conv_block = [MyConv2D(filters=256, kernel_size=[3, 5], dilation_rate=2) for _ in range(4)]
        self.aspp = ASPP(256, gird_size)
        self.conv1x1 = MyConv2D(64, kernel_size=1)
        self.output_conv = MyConv2D(num_class, 1, activation=False, apply_batchnorm=False)
        self.concat = Concatenate()

    def call(self, inputs, training=None, training_embedding=None, mask=None):
        x = inputs

        for conv in self.conv_block:
            x = conv(x, training=training)

        short_cut = x

        for conv in self.dilated_conv_block:
            x = conv(x, training=training)

        x = self.aspp(x, training=training)
        x = self.concat([x, short_cut])
        x = self.conv1x1(x, training=training)
        x = self.output_conv(x, training=training)

        return x


class ASPP(Layer):
    def __init__(self, filters, gird_size, name='aspp-layer', **kwargs):
        super(ASPP, self).__init__(name=name, **kwargs)
        self.conv = MyConv2D(filters=filters, kernel_size=1)
        self.upsampling = UpSampling2D(gird_size)
        self.dilated_conv_block1 = MyConv2D(filters=filters, kernel_size=1)
        self.dilated_conv_block4 = MyConv2D(filters=filters, kernel_size=[3, 5], dilation_rate=4)
        self.dilated_conv_block8 = MyConv2D(filters=filters, kernel_size=[3, 5], dilation_rate=8)
        self.dilated_conv_block16 = MyConv2D(filters=filters, kernel_size=[3, 5], dilation_rate=16)
        self.concat = Concatenate()
        self.conv1x1 = MyConv2D(filters, kernel_size=1)

    def call(self, inputs, training=None, **kwargs):
        image_features = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        image_features = self.conv(image_features, training=training)
        image_features = self.upsampling(image_features)

        dilated_conv_block1 = self.dilated_conv_block1(inputs, training=training)
        dilated_conv_block4 = self.dilated_conv_block4(inputs, training=training)
        dilated_conv_block8 = self.dilated_conv_block8(inputs, training=training)
        dilated_conv_block16 = self.dilated_conv_block16(inputs, training=training)

        x = self.concat(
            [image_features, dilated_conv_block1, dilated_conv_block4, dilated_conv_block8, dilated_conv_block16])

        x = self.conv1x1(x, training=training)

        return x


class BiLSTMClassifier(tf.keras.Model):
    def __init__(self, num_class, name='rnn-classifier', **kwargs):
        super(BiLSTMClassifier, self).__init__(name=name, **kwargs)
        self.rnn1 = Bidirectional(
            GRU(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.rnn2 = Bidirectional(
            GRU(num_class, return_sequences=True, activation='softmax', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.crf = CRF(num_class)

    def call(self, inputs, training=None, training_embedding=None, mask=None):
        # input shape = [batch_size, word_size, char_size]

        x = self.rnn1(inputs)  # shape = [batch_size, word_size, num_class]
        x = self.rnn2(x)  # shape = [batch_size, word_size, num_class]
        x = self.crf(x)

        return x

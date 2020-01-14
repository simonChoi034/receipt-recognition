import tensorflow as tf
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dense

from model.resnet import Resnet18


class Classifier(tf.keras.Model):
    def __init__(self, num_class, name='classifier', **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.resnet = Resnet18()
        self.reshape = Reshape((-1, 128))
        self.rnn_1 = Bidirectional(LSTM(128))
        self.rnn_2 = Bidirectional(LSTM(256))
        self.dense = Dense(num_class, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.resnet(inputs)
        x = self.reshape(x)  # shape = (None, None, 128)
        x = self.rnn_1(x)  # shape = (None, None, 128)
        x = self.rnn_2(x)  # shape = (None, None, 256)
        x = self.dense(x)  # shape = (None, num_class)

        return x

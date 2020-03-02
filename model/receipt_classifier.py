import tensorflow as tf
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dense, GRU, Embedding, TimeDistributed, RepeatVector

from model.resnet import Resnet18


# image classifier
class ResnetLSTMClassifier(tf.keras.Model):
    def __init__(self, num_class, name='resnet-lstm-classifier', **kwargs):
        super(ResnetLSTMClassifier, self).__init__(name=name, **kwargs)
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


class RNNClassifier(tf.keras.Model):
    def __init__(self, num_class, name='rnn-classifier', **kwargs):
        super(RNNClassifier, self).__init__(name=name, **kwargs)
        self.rnn1 = Bidirectional(
            GRU(64, return_sequences=True, recurrent_initializer='glorot_uniform'))
        self.rnn2 = Bidirectional(
            GRU(128, return_sequences=True, recurrent_initializer='glorot_uniform'))
        self.rnn3 = Bidirectional(
            GRU(256, return_sequences=True, recurrent_initializer='glorot_uniform'))
        self.dense = TimeDistributed(Dense(num_class))

    def call(self, inputs, training=None, mask=None):
        # input shape = [batch_size, word_size, 64]
        x = self.rnn1(inputs)  # shape = [batch_size, word_size, 128]
        x = self.rnn2(x)  # shape = [batch_size, word_size, 256]
        x = self.rnn3(x)  # shape = [batch_size, word_size, 512]
        x = self.dense(x)  # shape = [batch_size, word_size, num_class]

        return x


class WordEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, word_size, char_size,
                 name='word-embedding', **kwargs):
        super(WordEmbedding, self).__init__(name=name, **kwargs)
        self.word_size = word_size
        self.char_size = char_size
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder1 = GRU(128, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.encoder2 = GRU(64, return_sequences=False, recurrent_initializer='glorot_uniform')
        self.repeat_vector = RepeatVector(char_size)
        self.decoder1 = GRU(64, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.decoder2 = GRU(128, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.dense = TimeDistributed(Dense(vocab_size))

    def call(self, inputs, training=None, mask=None):
        # input shape: [batch_size, word_size, char_size]
        # input shape: [batch_size * word_size, char_size]
        x = tf.reshape(inputs, (-1, self.char_size))
        x = self.embedding(x)  # shape = [batch_size * word_size, char_size, embedding_dim]
        x = self.encoder1(x)  # shape = [batch_size * word_size, char_size, 128]
        x = self.encoder2(x)  # shape = [batch_size * word_size, 64]

        if not training:
            # output shape = [batch_size, word_size, 64]
            x = tf.reshape(x, (-1, self.word_size, 64))
            return x

        x = self.repeat_vector(x)  # shape = [batch_size * word_size, char_size, 64]
        x = self.decoder1(x)  # shape = [batch_size * word_size, char_size, 64]
        x = self.decoder2(x)  # shape = [batch_size * word_size, char_size, 128]
        x = self.dense(x)  # shape = [batch_size * word_size, char_size, vocab_size]

        x = tf.reshape(x, (-1, self.word_size, self.char_size, self.vocab_size))

        # output shape = [batch_size, word_size, char_size, vocab_size]
        return x

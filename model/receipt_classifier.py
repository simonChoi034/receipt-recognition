import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, GRU, Embedding, TimeDistributed, RepeatVector, \
    Layer, ZeroPadding2D
from tensorflow.keras.regularizers import l2

from model.crf import CRF
from model.layers import MyConv2D


# image classifier
class CnnBiLstmClassifier(tf.keras.Model):
    def __init__(self, num_class, vocab_size, embedding_dim, word_size, char_size, filters, kernel_size,
                 name='cnn-bilstm-classifier', **kwargs):
        super(CnnBiLstmClassifier, self).__init__(name=name, **kwargs)
        self.embedding = CnnEmbedding(vocab_size, embedding_dim, word_size, char_size, filters, kernel_size)
        self.rnn1 = Bidirectional(
            GRU(8, return_sequences=True, activation='tanh', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.rnn2 = Bidirectional(
            GRU(num_class, return_sequences=True, activation='softmax', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.crf = CRF(num_class)

    def call(self, inputs, training=None, mask=None):
        # input shape = [batch_size, word_size, char_size]
        x = self.embedding(inputs)  # shape = [batch_size, word_size, filter_size]

        x = self.rnn1(x)  # shape = [batch_size, word_size, x]
        x = self.rnn2(x)  # shape = [batch_size, word_size, num_class]
        x = self.crf(x)

        return x


class CnnEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, word_size, char_size, filters, kernel_size,
                 name='cnn-embedding', **kwargs):
        super(CnnEmbedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_size = word_size
        self.char_size = char_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.character_embedding = Embedding(vocab_size, embedding_dim)
        self.character_cnn1 = MyConv2D(
            filters=filters,
            kernel_size=(kernel_size, embedding_dim),
            padding='valid'
        )
        self.character_cnn2 = MyConv2D(
            filters=filters,
            kernel_size=(char_size, 1),
            padding='valid'
        )
        self.padding = ZeroPadding2D(padding=(1, 0))

    def call(self, inputs, **kwargs):
        # shape = [batch_size * word_size, char_size]
        x = tf.reshape(inputs, (-1, self.char_size))
        # shape = [batch_size * word_size, char_size, embedding_dim]
        x = self.character_embedding(x)
        # shape = [batch_size * word_size, char_size, embedding_dim, 1]
        x = tf.expand_dims(x, -1)
        # shape = [batch_size * word_size, char_size, 1, filter_size]
        x = self.padding(x)
        x = self.character_cnn1(x)
        # shape = [batch_size * word_size, 1, 1, filter_size]
        x = self.character_cnn2(x)
        x = tf.reshape(x, (-1, self.word_size, self.filters))

        return x


class RNNClassifier(tf.keras.Model):
    def __init__(self, num_class, vocab_size, embedding_dim, word_size, char_size, name='rnn-classifier', **kwargs):
        super(RNNClassifier, self).__init__(name=name, **kwargs)
        self.embedding = WordEmbedding(vocab_size, embedding_dim, word_size, char_size)
        self.rnn1 = Bidirectional(
            GRU(8, return_sequences=True, activation='tanh', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.rnn2 = Bidirectional(
            GRU(num_class, return_sequences=True, activation='softmax', kernel_regularizer=l2(),
                recurrent_regularizer=l2(), dropout=0.2,
                recurrent_dropout=0.2), merge_mode='sum')
        self.crf = CRF(num_class)

    def call(self, inputs, training=None, training_embedding=None, mask=None):
        # input shape = [batch_size, word_size, char_size]
        x = self.embedding(inputs, training=training_embedding)

        # training embedding layer
        if training_embedding:
            return x

        x = self.rnn1(x)  # shape = [batch_size, word_size, num_class]
        x = self.rnn2(x)  # shape = [batch_size, word_size, num_class]
        x = self.crf(x)

        return x


class WordEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, word_size, char_size,
                 name='word-embedding', **kwargs):
        super(WordEmbedding, self).__init__(name=name, **kwargs)
        self.word_size = word_size
        self.char_size = char_size
        self.vocab_size = vocab_size
        self.encode_dim = 16
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder1 = LSTM(self.encode_dim, return_sequences=False, recurrent_initializer='glorot_uniform')
        self.repeat_vector = RepeatVector(char_size)
        self.decoder1 = LSTM(self.encode_dim, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.dense = TimeDistributed(Dense(vocab_size))

    def call(self, inputs, training=None, mask=None):
        # input shape: [batch_size, word_size, char_size]
        # input shape: [batch_size * word_size, char_size]
        x = tf.reshape(inputs, (-1, self.char_size))
        x = self.embedding(x)  # shape = [batch_size * word_size, char_size, embedding_dim]
        x = self.encoder1(x)  # shape = [batch_size * word_size, char_size, encode_dim]

        if not training:
            # output shape = [batch_size, word_size, encode_dim]
            x = tf.reshape(x, (-1, self.word_size, self.encode_dim))
            return x

        x = self.repeat_vector(x)  # shape = [batch_size * word_size, char_size, encode_dim]
        x = self.decoder1(x)  # shape = [batch_size * word_size, char_size, encode_dim]
        x = self.dense(x)  # shape = [batch_size * word_size, char_size, vocab_size]

        x = tf.reshape(x, (-1, self.word_size, self.char_size, self.vocab_size))

        # output shape = [batch_size, word_size, char_size, vocab_size]
        return x

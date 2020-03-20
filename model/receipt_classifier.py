import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, GRU, Embedding, TimeDistributed, RepeatVector, \
    Layer, ZeroPadding2D, Dropout, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.regularizers import l2

from model.crf import CRF
from model.layers import MyConv2D


class GridClassifier(tf.keras.Model):
    def __init__(self, num_class, vocab_size, embedding_dim, gird_size, char_size, name='cnn-classifier', **kwargs):
        super(GridClassifier, self).__init__(name=name, **kwargs)
        self.embedding = GridBiLstmEmbedding(vocab_size, embedding_dim, gird_size, char_size)
        self.conv_block = [MyConv2D(filters=64, kernel_size=[3, 5]) for _ in range(2)]
        self.dilated_conv_block = [MyConv2D(filters=64, kernel_size=[3, 5], dilation_rate=2) for _ in range(2)]
        self.aspp = ASPP(64, gird_size)
        self.conv1x1 = MyConv2D(64, kernel_size=1)
        self.output_conv = MyConv2D(num_class, 1)
        self.concat = Concatenate()

    def call(self, inputs, training=None, training_embedding=None, mask=None):
        x = self.embedding(inputs, training=training_embedding)

        if training_embedding:
            return x

        for conv in self.conv_block:
            x = conv(x)

        for conv in self.dilated_conv_block:
            x = conv(x)

        short_cut = x
        x = self.aspp(x)
        x = self.concat([x, short_cut])
        x = self.conv1x1(x)
        x = self.output_conv(x)

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

    def call(self, inputs, **kwargs):
        image_features = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        image_features = self.conv(image_features)
        image_features = self.upsampling(image_features)

        dilated_conv_block1 = self.dilated_conv_block1(inputs)
        dilated_conv_block4 = self.dilated_conv_block4(inputs)
        dilated_conv_block8 = self.dilated_conv_block8(inputs)
        dilated_conv_block16 = self.dilated_conv_block16(inputs)

        x = self.concat(
            [image_features, dilated_conv_block1, dilated_conv_block4, dilated_conv_block8, dilated_conv_block16])

        x = self.conv1x1(x)

        return x


class GridBiLstmEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, grid_size, char_size, name='cnn-bilstm-embedding', **kwargs):
        super(GridBiLstmEmbedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.char_size = char_size
        self.encode_dim = 32
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder1 = LSTM(self.encode_dim, return_sequences=False, recurrent_initializer='glorot_uniform')
        self.repeat_vector = RepeatVector(char_size)
        self.decoder1 = LSTM(self.encode_dim, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.dense = TimeDistributed(Dense(vocab_size))

    def call(self, inputs, training=None, mask=None):
        # input shape: [batch_size, grid_size[0], grid_size[1], char_size]
        # input shape: [batch_size * grid_size[0] * grid_size[1], char_size]
        x = tf.reshape(inputs, (-1, self.char_size))
        x = self.embedding(x)  # shape = [batch_size * grid_size[0] * grid_size[1], char_size, embedding_dim]
        x = self.encoder1(x)  # shape = [batch_size * grid_size[0] * grid_size[1], encode_dim]

        if not training:
            # output shape = [batch_size, grid_size[0] * grid_size[1], encode_dim]
            x = tf.reshape(x, (-1, self.grid_size[0], self.grid_size[1], self.encode_dim))
            return x

        x = self.repeat_vector(x)  # shape = [batch_size * grid_size[0] * grid_size[1], char_size, encode_dim]
        x = self.decoder1(x)  # shape = [batch_size * grid_size[0] * grid_size[1], char_size, encode_dim]
        x = self.dense(x)  # shape = [batch_size * grid_size[0] * grid_size[1], char_size, vocab_size]

        x = tf.reshape(x, (-1, self.grid_size[0], self.grid_size[1], self.char_size, self.vocab_size))

        # output shape = [batch_size, grid_size[0], grid_size[1], char_size, vocab_size]
        return x


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
        self.max_pool = MaxPool2D(pool_size=(char_size, 1))
        self.padding = ZeroPadding2D(padding=(1, 0))
        self.dropout = Dropout(0.5)

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
        x = self.max_pool(x)
        # shape = [batch_size * word_size, filter_size]
        x = tf.squeeze(x)
        x = self.dropout(x)
        # shape = [batch_size, word_size, filter_size]
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

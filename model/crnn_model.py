import tensorflow as tf
from tensorflow.keras.regularizers import l2

mFILTERS = [64,256,512]

class CRNN(tf.keras.Model):


    def __init__(self, num_classes, training):
        super(CRNN, self).__init__()

        kernel_initializer = tf.random_normal_initializer(0., 0.05)
        bias_initializer = tf.constant_initializer(value=0.0)
        
        # cnn part
        self.conv1 = tf.keras.layers.Conv2D(filters=mFILTERS[0], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=mFILTERS[0], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=mFILTERS[1], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=training)

        self.conv4 = tf.keras.layers.Conv2D(filters=mFILTERS[1], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv5 = tf.keras.layers.Conv2D(filters=mFILTERS[2], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.bn5 = tf.keras.layers.BatchNormalization(trainable=training)

        self.conv6 = tf.keras.layers.Conv2D(filters=mFILTERS[2], kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv7 = tf.keras.layers.Conv2D(filters=mFILTERS[2], kernel_size=(2, 2), padding="valid", activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)

        # rnn part
        self.lstm_fw_cell_1 = tf.keras.layers.LSTM(mFILTERS[1], return_sequences=True)
        self.lstm_bw_cell_1 = tf.keras.layers.LSTM(mFILTERS[1], go_backwards=True, return_sequences=True)
        self.birnn1 = tf.keras.layers.Bidirectional(layer=self.lstm_fw_cell_1, backward_layer=self.lstm_bw_cell_1)

        self.lstm_fw_cell_2 = tf.keras.layers.LSTM(mFILTERS[1], return_sequences=True)
        self.lstm_bw_cell_2 = tf.keras.layers.LSTM(mFILTERS[1], go_backwards=True, return_sequences=True)
        self.birnn2 = tf.keras.layers.Bidirectional(layer=self.lstm_fw_cell_2, backward_layer=self.lstm_bw_cell_2)

        self.dense = tf.keras.layers.Dense(num_classes,activation='relu', kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)  # number of classes + 1 blank char

    def call(self, input, training = False):
        x = self.conv1(input, training=training)
        x = self.pool1(x)

        x = self.conv2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x, training=training)
        x = self.bn3(tf.cast(x, dtype=tf.float32), training=training)

        x = self.conv4(x, training=training)
        x = self.pool4(x)

        x = self.conv5(x, training=training)
        x = self.bn5(tf.cast(x, dtype=tf.float32), training=training)

        x = self.conv6(x, training=training)
        x = self.pool6(x)

        x = self.conv7(x, training=training)

        x = tf.reshape(x, [-1, x.shape[2], x.shape[3]])  # [BATCH, TIME, FILTERS]
        x = self.birnn1(x)
        x = self.birnn2(x)

        logits = self.dense(x)

        raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2)
        rnn_out = tf.transpose(logits, [1, 0, 2])

        return logits, raw_pred, rnn_out


import argparse
import datetime
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import confusion_matrix

from dataset.dataset import ClassifierDataset
from dataset.receipt.detector_dataset_generator import ReceiptClassifyGenerator
from model.receipt_classifier import WordEmbedding, RNNClassifier
from parameters import BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, LR_INIT

VOCAB_SIZE = 128
WORD_SIZE = 250
CHAR_SIZE = 100
EMBEDDING_DIM = 256
NUM_CLASS = 7
CLASS_NAME = ["Don't care", "Merchant Name", "Merchant Phone Number", "Merchant Address", "Transaction Date",
              "Transaction Time", "Total"]

try:
    tf.enable_eager_execution()
except:
    pass

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

embedding_layer = WordEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    word_size=WORD_SIZE,
    char_size=CHAR_SIZE
)
model = RNNClassifier(num_class=NUM_CLASS)
optimizer = tf.keras.optimizers.Adam(lr=LR_INIT, clipvalue=0.5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
accuracy = Accuracy()

# checkpoint manager
embedding_layer_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=embedding_layer)
embedding_layer_manager = tf.train.CheckpointManager(embedding_layer_ckpt, './checkpoints/embedding_layer_train.tf',
                                                     max_to_keep=5)
model_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
model_manager = tf.train.CheckpointManager(model_ckpt, './checkpoints/receipt_classifier_train.tf', max_to_keep=5)

# tensorboard config
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
embedding_layer_train_log_dir = 'logs/embedding_layer/' + current_time + '/train'
receipt_classifier_train_log_dir = 'logs/receipt_classifier/' + current_time + '/train'


@tf.function
def train_embedding_layer_one_step(x, y):
    with tf.GradientTape() as tape:
        pred = embedding_layer(x, training=True)
        loss = loss_fn(y_true=y, y_pred=pred)

    grads = tape.gradient(loss, embedding_layer.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, embedding_layer.trainable_variables))

    return loss, pred


def train_embedding_layer(dataset):
    # setup tensorboard
    train_summary_writer = tf.summary.create_file_writer(embedding_layer_train_log_dir)

    # restore checkpoint
    embedding_layer_ckpt.restore(embedding_layer_manager.latest_checkpoint)

    if embedding_layer_manager.latest_checkpoint:
        print("Restored from {}".format(embedding_layer_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for data in dataset:
        loss, pred = train_embedding_layer_one_step(data['word_list'], data['word_list'])
        pred = np.argmax(pred.numpy(), axis=-1)  # shape = [batch_size, word_size, char_size]

        embedding_layer_ckpt.step.assign_add(1)

        if int(embedding_layer_ckpt.step) % 100 == 0:
            # tensorboard logging
            y_true_string = ascii_to_string(data['word_list'].numpy()[0][0])
            y_pred_string = ascii_to_string(pred[0][0])
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=int(embedding_layer_ckpt.step))
                tf.summary.text("Prediction", y_pred_string, step=int(embedding_layer_ckpt.step))
                tf.summary.text("Ground Truth", y_true_string, step=int(embedding_layer_ckpt.step))

            save_path = embedding_layer_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(embedding_layer_ckpt.step), save_path))
            print("training loss {:1.5f}".format(loss.numpy()))

        if loss < 2e-3:
            embedding_layer.save('./saved_model/embedding_layer')
            print("Training finished")
            print("Final loss {:1.5f}".format(loss.numpy()))
            return


@tf.function
def train_classifier_one_step(x, y):
    with tf.GradientTape() as tape:
        embedding = embedding_layer(x)
        pred = model(embedding)
        loss = loss_fn(y_true=y, y_pred=pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables))

    return loss, pred


def train_classifier(dataset):
    # load embedding layer
    global embedding_layer
    embedding_layer = tf.keras.models.load_model('saved_model/embedding_layer')
    embedding_layer.trainable = False

    # setup tensorboard
    train_summary_writer = tf.summary.create_file_writer(receipt_classifier_train_log_dir)

    # restore checkpoint
    model_ckpt.restore(model_manager.latest_checkpoint)

    if model_manager.latest_checkpoint:
        print("Restored from {}".format(model_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # set the word embedding layer to non-trainable
    embedding_layer.trainable = False

    for data in dataset:
        loss, pred = train_classifier_one_step(data['word_list'], data['label'])
        pred = np.argmax(pred, axis=-1)

        tf.print(loss)

        model_ckpt.step.assign_add(1)

        if int(model_ckpt.step) % 100:
            accuracy.update_state(y_true=data['label'], y_pred=pred)
            confusion_matrix = create_confusion_matrix(y_true=data['label'].numpy(), y_pred=pred)

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=int(model_ckpt.step))
                tf.summary.scalar("Accuracy", accuracy.result().numpy(), step=int(model_ckpt.step))
                tf.summary.image("Confusion Matrix", confusion_matrix, step=int(model_ckpt.step))

        if loss < 1e-3:
            model.save('./saved_model/receipt_classifier')
            print("Training finished")
            print("Final loss {:1.5f}".format(loss.numpy()))
            return


def create_confusion_matrix(y_true, y_pred):
    y_true = np.reshape(y_true, (-1)).astype(int)
    y_pred = np.reshape(y_pred, (-1)).astype(int)
    con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(NUM_CLASS), normalize='true')
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_norm = np.nan_to_num(con_mat_norm)
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=CLASS_NAME,
                              columns=CLASS_NAME)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def ascii_to_string(ascii_array):
    return ''.join(chr(i) for i in ascii_array)


def main(args):
    dataset_dir = args.dir[0:-1] if args.dir[-1] == '/' else args.dir
    receipt_generator = ReceiptClassifyGenerator(
        dataset_dir=dataset_dir,
        vocab_size=VOCAB_SIZE,
        word_size=WORD_SIZE,
        char_size=CHAR_SIZE
    )

    receipt_generator.set_dataset_info()

    dataset_generator = ClassifierDataset(
        generator=receipt_generator,
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )

    dataset = dataset_generator.create_dataset()

    if args.emb:
        train_embedding_layer(dataset=dataset)
    else:
        train_classifier(dataset=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('-e', '--emb', action='store_true', help='Train character embedding layer')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-d', '--dir', help='Directory of dataset', required=True)
    parser.add_argument('-s', action='store_true', help='Shut down vm after training stop')
    args = parser.parse_args()

    main(args)

import argparse
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from dataset.dataset import ClassifierDataset
from dataset.receipt.detector_dataset_generator import ReceiptClassifyGenerator
from model.receipt_classifier import WordEmbedding, RNNClassifier
from parameters import BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, LR_INIT

VOCAB_SIZE = 128
WORD_SIZE = 250
CHAR_SIZE = 100
EMBEDDING_DIM = 256
NUM_CLASS = 5

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
    # restore checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=embedding_layer)
    manager = tf.train.CheckpointManager(ckpt, './checkpoints/embedding_layer_train.tf', max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/embedding_layer/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for data in dataset:
        loss, pred = train_embedding_layer_one_step(data['word_list'], data['word_list'])
        pred = np.argmax(pred.numpy(), axis=-1)  # shape = [batch_size, word_size, char_size]

        ckpt.step.assign_add(1)

        if int(ckpt.step) % 10 == 0:
            # tensorboard logging
            y_true_string = ascii_to_string(data['word_list'].numpy()[0][0])
            y_pred_string = ascii_to_string(pred[0][0])
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=int(ckpt.step))
                tf.summary.text("Prediction", y_pred_string, step=int(ckpt.step))
                tf.summary.text("Ground Truth", y_true_string, step=int(ckpt.step))

            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("training loss {:1.2f}".format(loss.numpy()))

            if loss < 1e-3:
                print("Training finished")
                print("Final loss {:1.3f}".format(loss.numpy()))


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
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './checkpoints/receipt_classifier_train.tf', max_to_keep=5)

    # restore checkpoint
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # set the word embedding layer to non-trainable
    embedding_layer.trainable = False

    for data in dataset:
        loss, pred = train_classifier_one_step(data['word_list'], data['label'])
        print(loss)


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

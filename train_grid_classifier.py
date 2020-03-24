import argparse
import datetime
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from dataset.dataset import ClassifierDataset
from dataset.receipt.detector_dataset_generator import GridReceiptClassifyGenerator
from model.receipt_classifier import GridClassifier
from parameters import BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, LR_INIT, LR_END

CHAR_SIZE_OPTIONS = [50, 100, 150, 200, 300]

VOCAB_SIZE = 128
WORD_SIZE = 250
CHAR_SIZE = 50
EMBEDDING_DIM = 128
WARMUP_EPOCHS = 10
TRAIN_EPOCHS = 3000
NUM_CLASS = 5
GRID_SIZE = [50, 50]
CLASS_NAME = ["Don't care", "Merchant Name", "Merchant Address", "Transaction Date", "Total"]

train_config = {
    'warmup_steps': WARMUP_EPOCHS,
    'total_steps': TRAIN_EPOCHS
}

try:
    tf.enable_eager_execution()
except:
    pass

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = GridClassifier(
    num_class=NUM_CLASS,
    gird_size=GRID_SIZE
)

optimizer = tf.keras.optimizers.Adam(lr=LR_INIT)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)

# checkpoint manager
model_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
model_manager = tf.train.CheckpointManager(model_ckpt, './checkpoints/grid_receipt_classifier_train.tf', max_to_keep=5)

# tensorboard config
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
receipt_classifier_train_log_dir = 'logs/receipt_classifier/grid/' + current_time + '/train'
receipt_classifier_val_log_dir = 'logs/receipt_classifier/grid/' + current_time + '/val'


def update_learning_rate(step):
    global_steps = step
    warmup_steps = train_config['warmup_steps']
    total_steps = train_config['total_steps']
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )

    optimizer.lr.assign(float(lr))


@tf.function
def model_validation(x, y):
    pred = model(x, training=True)
    loss = loss_fn(y_true=y, y_pred=pred)
    loss = tf.reduce_mean(loss)

    return loss, pred


@tf.function
def train_classifier_one_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y_true=y, y_pred=pred)
        loss = tf.reduce_mean(loss)
        regularization_loss = tf.reduce_sum(model.losses)
        total_loss = loss + regularization_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables))

    return loss, pred


def train_classifier(train_dataset, val_dataset):
    # setup tensorboard
    train_summary_writer = tf.summary.create_file_writer(receipt_classifier_train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(receipt_classifier_val_log_dir)

    if model_manager.latest_checkpoint:
        # restore checkpoint
        model_ckpt.restore(model_manager.latest_checkpoint)
        print("Restored from {}".format(model_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for data in train_dataset:
        train_loss, preds = train_classifier_one_step(data['word_list'], data['label'])

        update_learning_rate(int(model_ckpt.step))

        model_ckpt.step.assign_add(1)

        if int(model_ckpt.step) % 100 == 0:
            logging(train_summary_writer, preds, data['label'], train_loss)

            # validation set
            data_val = next(iter(val_dataset))
            val_loss, preds = model_validation(data_val['word_list'], data_val['label'])

            logging(val_summary_writer, preds, data_val['label'], val_loss)

            save_path = model_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(model_ckpt.step), save_path))
            print("training loss {:1.5f}".format(train_loss.numpy()))

        if int(model_ckpt.step) >= train_config['total_steps']:
            model.save('./saved_model/receipt_classifier')
            print("Training finished")
            print("Final loss {:1.5f}".format(train_loss.numpy()))
            return


def logging(writer, preds, labels, loss):
    preds = np.argmax(preds.numpy(), axis=-1)
    # training set
    confusion_matrix = create_confusion_matrix(y_true=labels, y_pred=preds)
    report, mean_precision, mean_recall, mean_f1 = create_classification_report(y_true=labels, y_pred=preds)

    with writer.as_default():
        tf.summary.scalar("lr", optimizer.lr, step=int(model_ckpt.step))
        tf.summary.scalar("loss", loss, step=int(model_ckpt.step))
        for name in CLASS_NAME:
            tf.summary.scalar("{} precision".format(name), report[name]['precision'], step=int(model_ckpt.step))
            tf.summary.scalar("{} recall".format(name), report[name]['recall'], step=int(model_ckpt.step))
            tf.summary.scalar("{} f1-score".format(name), report[name]['f1-score'], step=int(model_ckpt.step))
        tf.summary.scalar("Mean precision", mean_precision, step=int(model_ckpt.step))
        tf.summary.scalar("Mean recall", mean_recall, step=int(model_ckpt.step))
        tf.summary.scalar("Mean f1-score", mean_f1, step=int(model_ckpt.step))
        tf.summary.image("Confusion Matrix", confusion_matrix, step=int(model_ckpt.step))


def create_classification_report(y_true, y_pred):
    y_true = np.reshape(y_true, (-1)).astype(int)
    y_pred = np.reshape(y_pred, (-1)).astype(int)
    report = classification_report(y_true, y_pred, labels=range(NUM_CLASS), digits=3, output_dict=True, zero_division=0,
                                   target_names=CLASS_NAME)
    mean_precision = np.mean([report[name]['precision'] for name in CLASS_NAME])
    mean_recall = np.mean([report[name]['recall'] for name in CLASS_NAME])
    mean_f1 = np.mean([report[name]['f1-score'] for name in CLASS_NAME])

    return report, mean_precision, mean_recall, mean_f1


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
    train_receipt_generator = GridReceiptClassifyGenerator(
        grid_size=GRID_SIZE,
        dataset_dir=dataset_dir,
        vocab_size=VOCAB_SIZE,
        word_size=WORD_SIZE,
        char_size=args.char_size,
        mode='train'
    )
    val_receipt_generator = GridReceiptClassifyGenerator(
        grid_size=GRID_SIZE,
        dataset_dir=dataset_dir,
        vocab_size=VOCAB_SIZE,
        word_size=WORD_SIZE,
        char_size=args.char_size,
        mode='val'
    )

    train_receipt_generator.set_dataset_info()
    val_receipt_generator.set_dataset_info()

    dataset_size = len(train_receipt_generator.grids)
    warmup_steps = WARMUP_EPOCHS * dataset_size // args.batch_size
    total_steps = TRAIN_EPOCHS * dataset_size // args.batch_size

    train_dataset_generator = ClassifierDataset(
        generator=train_receipt_generator,
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )
    val_dataset_generator = ClassifierDataset(
        generator=val_receipt_generator,
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )

    train_dataset = train_dataset_generator.create_dataset()
    val_dataset = val_dataset_generator.create_dataset()

    # reset training config
    set_training_config(warmup_steps, total_steps)
    train_classifier(train_dataset=train_dataset, val_dataset=val_dataset)


def set_training_config(warmup_steps, total_steps):
    train_config['warmup_steps'] = warmup_steps
    train_config['total_steps'] = total_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument(
        '-c',
        '--char_size',
        default=50,
        choices=CHAR_SIZE_OPTIONS,
        type=int,
        nargs='?',
        help='Enter the character embedding size'
    )
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-d', '--dir', help='Directory of dataset', required=True)
    parser.add_argument('-s', action='store_true', help='Shut down vm after training stop')
    args = parser.parse_args()

    main(args)
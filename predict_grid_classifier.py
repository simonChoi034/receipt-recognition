import argparse
import json
import chars2vec

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from model.receipt_classifier import GridClassifier

LR_INIT = 1e-4
VOCAB_SIZE = 128
WORD_SIZE = 250
CHAR_SIZE = 50
WARMUP_EPOCHS = 100
TRAIN_EPOCHS = 1500
NUM_CLASS = 5
GRID_SIZE = [64, 64]
CLASS_NAME = ["Don't care", "Merchant Name", "Merchant Address", "Transaction Date", "Total"]

# model config
c2v_model = chars2vec.load_model('eng_{}'.format(150))
model = GridClassifier(
    num_class=NUM_CLASS,
    gird_size=GRID_SIZE
)
optimizer = tf.keras.optimizers.Adam(lr=LR_INIT, clipnorm=10.0)
cross_entropy = SparseCategoricalCrossentropy(from_logits=True)
model_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
model_manager = tf.train.CheckpointManager(model_ckpt, './checkpoints/grid_receipt_classifier_train.tf', max_to_keep=5)
if model_manager.latest_checkpoint:
    # restore checkpoint
    model_ckpt.restore(model_manager.latest_checkpoint)
    print("Restored from {}".format(model_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# loss function
class_weights = [0.1, 1, 1.2, 0.8, 1.5]

class_ids = {
    "MerchantName": 1,
    "MerchantAddress": 2,
    "TransactionDate": 3,
    "Total": 4
}


def create_grid(document):
    text_grid = [['' for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
    input_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 150))
    label_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1]))
    for page in document['readResults']:
        width, height = page['width'], page['height']
        for line in page['lines']:
            for word in line['words']:
                # calculate grid index
                x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                x_min, x_max = min([x1, x2, x3, x4]), max([x1, x2, x3, x4])
                y_min, y_max = min([y1, y2, y3, y4]), max([y1, y2, y3, y4])
                w, h = x_max - x_min, y_max - y_min
                x_cen, y_cen = x_min + w / 2, y_min + h / 2
                x_cen, y_cen = x_cen / width, y_cen / height

                column_index = int(x_cen * GRID_SIZE[1])
                row_index = int(y_cen * GRID_SIZE[0])

                # encode text and class_id
                class_id = int(word['class'])
                text = [str(word['text'])]
                encoded_text = c2v_model.vectorize_words(text)

                text_grid[row_index][column_index] = word['text']
                input_grid[row_index][column_index] = encoded_text[0]
                label_grid[row_index][column_index] = class_id

    return text_grid, input_grid, label_grid


def pad_class_id(document):
    # initial all words class id to 0
    for page in document['readResults']:
        for line in page['lines']:
            for word in line['words']:
                word['class'] = 0

    # set class id for words
    document_results = document['documentResults']
    for document_result in document_results:
        class_fields = document_result['fields']
        for class_key, class_id in class_ids.items():
            if class_key in class_fields:
                for word_element in class_fields[class_key]['elements']:
                    word_element = word_element.split('/')
                    page_idx = int(word_element[2])
                    line_idx = int(word_element[4])
                    word_idx = int(word_element[6])
                    document['readResults'][page_idx]['lines'][line_idx]['words'][word_idx]['class'] = class_id

    return document


def read_file(file):
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        return data['analyzeResult']


def loss_fn(y_true, y_pred):
    # loss with class weights
    sample_weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
    return cross_entropy(y_true=y_true, y_pred=y_pred, sample_weight=sample_weights)


@tf.function
def predict(x, y):
    pred = model(x, training=True)
    loss = loss_fn(y_true=y, y_pred=pred)
    loss = tf.reduce_mean(loss)

    return loss, pred


def main(args):
    filename = args.dir
    data = read_file(filename)
    data = pad_class_id(data)
    text_grid, input_grid, label_grid = create_grid(data)

    input_grid = np.expand_dims(input_grid, 0)
    label_grid = np.expand_dims(label_grid, 0)

    loss, pred = predict(input_grid, label_grid)

    pred = np.reshape(np.argmax(pred.numpy(), axis=-1), (-1))
    # label = np.reshape(data_pred['label'], (-1))
    word_grid = np.reshape(text_grid, (-1))

    for i, class_name in enumerate(CLASS_NAME):
        if i != 0:
            index = np.where(pred == i)
            texts = word_grid[index]
            print("{}:".format(class_name))
            print(' '.join(texts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict classifier')
    parser.add_argument('-d', '--dir', help='Directory of filename', required=True)
    args = parser.parse_args()

    main(args)

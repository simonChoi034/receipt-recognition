import argparse

import cv2
import numpy as np
import tensorflow as tf

from model.crnn_model import CRNN
from recognizor_utils import params, char_dict, decode_to_text

model = CRNN(num_classes=params['NUM_CLASSES'], training=True)
model.load_weights('checkpoints/model_default/model_default')


def preprocess_input_image(image, height=params['INPUT_SIZE'][1], width=params['INPUT_SIZE'][0]):
    scale_rate = height / image.shape[0]
    tmp_new_width = int(scale_rate * image.shape[1])
    new_width = width if tmp_new_width > width else tmp_new_width
    image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)

    r, c = np.shape(image)
    if c > width:
        ratio = float(width) / c
        image = cv2.resize(image, (width, int(32 * ratio)))
    else:
        width_pad = width - image.shape[1]
        image = np.pad(image, pad_width=[(0, 0), (0, width_pad)], mode='constant', constant_values=0)

    image = image[:, :, np.newaxis]

    return np.expand_dims(image, 0).astype(float)


@tf.function
def predict(x):
    logits, raw_pred, rnn_out = model(x)
    return logits, raw_pred, rnn_out


def main(args):
    image_path = args.image
    image = cv2.imread(image_path, 0)
    image = preprocess_input_image(image)
    logits, raw_pred, rnn_out = predict(image)
    decoded_test, _ = tf.nn.ctc_greedy_decoder(rnn_out,  # logits.numpy().transpose((1, 0, 2)),
                                               sequence_length=[params['SEQ_LENGTH']],
                                               merge_repeated=True)
    decoded_test = tf.sparse.to_dense(decoded_test[0]).numpy()
    pre_ = [decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_test]
    print(pre_[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict recondition model')
    parser.add_argument('-i', '--image', help='Pathname of the image', required=True)
    args = parser.parse_args()

    main(args)

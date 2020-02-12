import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.yolov3 import YoloV3, output_bbox
from parameters import NUM_CLASS, IMAGE_SIZE

model = YoloV3(num_class=NUM_CLASS)
model.load_weights('./saved_model_weight/saved_weight')


@tf.function
def predict(image):
    pred_s, pred_m, pred_l = model(image)
    bboxes, scores, classes, valid_detections = output_bbox((pred_s, pred_m, pred_l))

    return bboxes, scores, classes, valid_detections


def read_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], preserve_aspect_ratio=True)
    img = tf.image.pad_to_bounding_box(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
    return img / 127.5 - 1


def main(args):
    image_path = args.image
    image = read_image(image_path)
    image = tf.expand_dims(image, axis=0)

    bboxes, scores, classes, valid_detections = predict(image)
    plot_bounding_box(image, bboxes, scores, valid_detections)


def plot_bounding_box(imgs, labels, scores, valid_detections):
    img = imgs.numpy()[0]
    label = labels.numpy()[0][:valid_detections.numpy()[0]]
    score = scores.numpy()[0][:valid_detections.numpy()[0]]

    # set random color
    colors = np.random.rand(100)
    cmap = plt.cm.RdYlBu_r
    c = cmap((np.array(colors) - np.amin(colors)) / (np.amax(colors) - np.amin(colors)))

    # normalize image to [0, 1]
    img = (img + 1) / 2
    img_h, img_w = img.shape[0], img.shape[1]

    # plot graph
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i, (bbox, score) in enumerate(zip(label, score)):
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1) * img_w
        h = abs(y2 - y1) * img_h
        x, y = x1 * img_w, y1 * img_h

        rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                  edgecolor='blue', facecolor='none')
        ax.annotate("{:1.2f}".format(score), (x, y - 4), color=c[i])
        ax.add_patch(rect)

    img_file = "./figure/predict_image.png"
    plt.savefig(img_file)
    plt.show()

    image = plt.imread(img_file)
    image = np.expand_dims(image, axis=0)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict detection model')
    parser.add_argument('-i', '--image', help='Pathname of the image', required=True)
    args = parser.parse_args()

    main(args)

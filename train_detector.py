import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from coco_text.dataset import Dataset, COCOGenerator
from model.yolov3 import YoloV3, yolo_loss

dataset_choice = ['coco_text']
IMAGE_SIZE = 416
BATCH_SIZE = 32
BUFFER_SIZE = 320
PREFETCH_SIZE = 5
LEARNING_RATE = 1e-3
NUM_CLASS = 2


def train_one_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        regularization_loss = tf.reduce_sum(model.losses)
        pred_loss = []
        for pred, y, loss_fn in zip(pred, y, loss_fn):
            pred_loss.append(loss_fn(y, pred))

        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))

        return total_loss


def train(model, dataset, optimizer):
    anchors = YoloV3.yolo_anchors
    anchor_masks = YoloV3.yolo_anchor_masks
    loss_fn = [yolo_loss(anchors[mask], num_class=NUM_CLASS) for mask in anchor_masks]

    for epoch, data in enumerate(dataset):
        tf.print("Epochs", epoch)
        loss = train_one_step(model, optimizer, loss_fn, data['image'], data['label'])
        tf.print("Loss: ", loss)

        if np.array(epoch) % 100 == 0:
            model.save_weights(
                './checkpoints/yolov3_train_{}.tf'.format(epoch))


def main(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = YoloV3(num_class=NUM_CLASS)
    optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

    if args.dataset == 'coco_text':
        # set up dataset config
        imgs_dir = args.dir[0:-1] if args.dir[-1] == '/' else args.dir
        coco_generator = COCOGenerator('./coco_text/cocotext.v2.json',
                                       imgs_dir,
                                       image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
                                       anchors=YoloV3.yolo_anchors,
                                       anchor_masks=YoloV3.yolo_anchor_masks
                                       )
        coco_generator.set_dataset_info()

        dataset_generator = Dataset(
            generator=coco_generator,
            image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            prefetch_size=PREFETCH_SIZE
        )
        dataset = dataset_generator.create_dataset()

        # train network
        train(model, dataset, optimizer)


def plot_bounding_box(img, label):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox in label:
        x, y, w, h, _ = bbox
        rect = mpatches.Rectangle((x * IMAGE_SIZE, y * IMAGE_SIZE), w * IMAGE_SIZE, h * IMAGE_SIZE, linewidth=2,
                                  edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument(
        '--dataset',
        default='coco_text',
        choices=dataset_choice,
        nargs='?',
        help='Enter the dataset'
    )
    parser.add_argument('-d', '--dir', help='Enter the directory of dataset', required=True)
    args = parser.parse_args()

    main(args)

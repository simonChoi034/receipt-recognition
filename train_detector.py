import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from coco_text.dataset import Dataset, COCOGenerator
from model.yolov3 import YoloV3, yolo_loss, yolo_anchors, yolo_anchor_masks, output_bbox
from parameters import dataset_choice, IMAGE_SIZE, BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, NUM_CLASS, LEARNING_RATE

try:
    tf.enable_eager_execution()
except:
    pass

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = YoloV3(num_class=NUM_CLASS)
optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)


def validation(x, y):
    # calculate loss from validation dataset
    pred_s, pred_m, pred_l = model(x)
    true_s, true_m, true_l = y
    regularization_loss = tf.reduce_sum(model.losses)
    pred_loss = yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)
    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

    # get bounding box
    bbox, objectness, class_probs, pred_box = output_bbox((pred_s, pred_m, pred_l))

    return total_loss, bbox, objectness, class_probs, pred_box


def train_one_step(x, y):
    with tf.GradientTape() as tape:
        pred_s, pred_m, pred_l = model(x, training=True)
        true_s, true_m, true_l = y
        regularization_loss = tf.reduce_sum(model.losses)

        pred_loss = yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)

        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables))

    return total_loss


def train(dataset_train, dataset_val):
    train_loss = []
    val_loss = []

    iterator_val = dataset_val.make_one_shot_iterator()

    for epoch, data in enumerate(dataset_train):
        loss = train_one_step(data['image'], data['label'])
        train_loss.append(loss)

        if np.array(epoch) % 200 == 0:
            tf.print("Epochs", epoch)
            # validation ever 100 epochs
            data_val = iterator_val.get_next()
            loss, bbox, objectness, class_probs, pred_box = validation(data_val['image'], data_val['label'])
            val_loss.append(loss)

            tf.print("Validation loss: ", loss)

            plot_bounding_box(data_val['image'].numpy()[0], bbox.numpy()[0])

            model.save_weights(
                './checkpoints/yolov3_train_{}.tf'.format(epoch))


def main(args):
    if args.dataset == 'coco_text':
        # set up dataset config
        imgs_dir = args.dir[0:-1] if args.dir[-1] == '/' else args.dir
        coco_train_generator = COCOGenerator('./coco_text/cocotext.v2.json',
                                             imgs_dir,
                                             mode='train',
                                             batch_size=BATCH_SIZE,
                                             image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
                                             anchors=yolo_anchors,
                                             anchor_masks=yolo_anchor_masks
                                             )
        coco_val_generator = COCOGenerator('./coco_text/cocotext.v2.json',
                                           imgs_dir,
                                           mode='val',
                                           batch_size=BATCH_SIZE,
                                           image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
                                           anchors=yolo_anchors,
                                           anchor_masks=yolo_anchor_masks
                                           )
        coco_train_generator.set_dataset_info()
        coco_val_generator.set_dataset_info()

        dataset_train_generator = Dataset(
            generator=coco_train_generator,
            image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            prefetch_size=PREFETCH_SIZE
        )
        dataset_val_generator = Dataset(
            generator=coco_val_generator,
            image_input_size=[IMAGE_SIZE, IMAGE_SIZE],
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            prefetch_size=PREFETCH_SIZE
        )
        dataset_train = dataset_train_generator.create_dataset()
        dataset_val = dataset_val_generator.create_dataset()

        # train network
        train(dataset_train, dataset_val)


def plot_bounding_box(img, label):
    # set random color
    colors = np.random.rand(500)
    cmap = plt.cm.RdYlBu_r
    c = cmap((np.array(colors) - np.amin(colors)) / (np.amax(colors) - np.amin(colors)))

    # normalize image to [0, 1]
    img = (img + 1) / 2

    # plot graph
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i, bbox in enumerate(label):
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1) * IMAGE_SIZE
        h = abs(y2 - y1) * IMAGE_SIZE
        x, y = x1 * IMAGE_SIZE, y1 * IMAGE_SIZE

        rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                  edgecolor=c[i], facecolor='none')
        ax.add_patch(rect)
    plt.show()


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

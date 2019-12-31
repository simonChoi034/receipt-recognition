import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from dataset.coco_text.dataset_generator import COCOGenerator
from dataset.receipt.dataset_generator import ReceiptGenerator
from dataset.dataset import Dataset
from model.yolov3 import YoloV3, yolo_loss, yolo_anchors, yolo_anchor_masks, output_bbox
from parameters import dataset_choice, IMAGE_SIZE, BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, NUM_CLASS, LEARNING_RATE

try:
    tf.enable_eager_execution()
except:
    pass

checkpoint_dir = './checkpoints/yolov3_train.tf'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = YoloV3(num_class=NUM_CLASS)
optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)


@tf.function
def validation(x, y):
    # calculate loss from validation dataset
    pred_s, pred_m, pred_l = model(x)
    true_s, true_m, true_l = y
    pred_loss = yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)

    # get bounding box
    bbox, objectiveness, class_probs, pred_box = output_bbox((pred_s, pred_m, pred_l))

    return pred_loss, bbox, objectiveness, class_probs, pred_box


@tf.function
def train_one_step(x, y):
    with tf.GradientTape() as tape:
        pred_s, pred_m, pred_l = model(x, training=True)
        true_s, true_m, true_l = y
        regularization_loss = tf.reduce_sum(model.losses)

        pred_loss = yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)

        total_loss = pred_loss + regularization_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables))

    return pred_loss


def train(dataset_train, dataset_val):
    # restore checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_loss = []
    val_loss = []

    iterator_val = dataset_val.make_one_shot_iterator()

    for data in dataset_train:
        loss = train_one_step(data['image'], data['label'])
        train_loss.append(loss)

        ckpt.step.assign_add(1)

        if int(ckpt.step) % 100 == 0:
            tf.print("Steps: ", int(ckpt.step))
            # validation ever 100 epochs
            data_val = iterator_val.get_next()
            loss, bbox, objectiveness, class_probs, pred_box = validation(data_val['image'], data_val['label'])
            val_loss.append(loss)

            tf.print("Train loss: ", train_loss[-1])
            tf.print("Validation loss: ", val_loss[-1])

            plot_bounding_box(data_val['image'].numpy()[0], bbox.numpy()[0])

            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))


def main(args):
    dataset_dir = args.dir[0:-1] if args.dir[-1] == '/' else args.dir
    if args.dataset == 'coco_text':
        # set up dataset config
        train_generator = COCOGenerator('dataset/coco_text/cocotext.v2.json',
                                        dataset_dir,
                                        mode='train',
                                        batch_size=args.batch_size,
                                        image_input_size=[args.image_size, args.image_size],
                                        anchors=yolo_anchors,
                                        anchor_masks=yolo_anchor_masks
                                        )
        val_generator = COCOGenerator('dataset/coco_text/cocotext.v2.json',
                                      dataset_dir,
                                      mode='val',
                                      batch_size=args.batch_size,
                                      image_input_size=[args.image_size, args.image_size],
                                      anchors=yolo_anchors,
                                      anchor_masks=yolo_anchor_masks
                                      )
    else:
        train_generator = ReceiptGenerator(dataset_dir,
                                           batch_size=args.batch_size,
                                           image_input_size=[args.image_size, args.image_size],
                                           anchors=yolo_anchors,
                                           anchor_masks=yolo_anchor_masks
                                           )
        val_generator = ReceiptGenerator(dataset_dir,
                                         batch_size=args.batch_size,
                                         image_input_size=[args.image_size, args.image_size],
                                         anchors=yolo_anchors,
                                         anchor_masks=yolo_anchor_masks
                                         )

    train_generator.set_dataset_info()
    val_generator.set_dataset_info()

    dataset_train_generator = Dataset(
        generator=train_generator,
        image_input_size=[args.image_size, args.image_size],
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )
    dataset_val_generator = Dataset(
        generator=val_generator,
        image_input_size=[args.image_size, args.image_size],
        batch_size=args.batch_size,
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
    img_h, img_w = img.shape[0], img.shape[1]

    # plot graph
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i, bbox in enumerate(label):
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1) * img_w
        h = abs(y2 - y1) * img_h
        x, y = x1 * img_w, y1 * img_h

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
    parser.add_argument('-d', '--dir', help='Directory of dataset', required=True)
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-i', '--image_size', default=IMAGE_SIZE, help='Reshape size of the image')
    args = parser.parse_args()

    main(args)

import argparse
import datetime
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset.coco_text.detector_dataset_generator import COCOGenerator
from dataset.dataset import DetectorDataset
from dataset.receipt.detector_dataset_generator import ReceiptGenerator
from dataset.synthtext.detector_dataset_generator import SynthTextGenerator
from model.yolov3 import YoloV3, yolo_loss, yolo_anchors, yolo_anchor_masks, output_bbox, precision, recall, mAP
from parameters import dataset_choice, IMAGE_SIZE, BATCH_SIZE, BUFFER_SIZE, PREFETCH_SIZE, NUM_CLASS, LR_INIT, LR_END, \
    WARMUP_EPOCHS, TRAIN_EPOCHS

try:
    tf.enable_eager_execution()
except:
    pass

# runtime config for training
train_config = {
    'batch_size': BATCH_SIZE,
    'dataset_size': 1,
    'warmup_steps': WARMUP_EPOCHS,
    'total_steps': TRAIN_EPOCHS
}

checkpoint_dir = './checkpoints/yolov3_2.0_train.tf'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = YoloV3(num_class=NUM_CLASS)
optimizer = tf.keras.optimizers.Adam(lr=LR_INIT, clipvalue=0.5)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

# setup tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
val_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)


@tf.function
def validation(x, y):
    # calculate loss from validation dataset
    pred_s, pred_m, pred_l = model(x)
    true_s, true_m, true_l = y
    pred_loss = yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)

    # get bounding box
    bboxes, scores, classes, valid_detections = output_bbox((pred_s, pred_m, pred_l))

    return pred_loss, bboxes, scores, classes, valid_detections


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

    # adaptive learning rate for each step
    update_learning_rate()

    return pred_loss


def update_learning_rate():
    global_steps = int(ckpt.step)
    warmup_steps = train_config['warmup_steps']
    total_steps = train_config['total_steps']
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )

    optimizer.lr.assign(float(lr))


def metrics_logging(writer, data, generator):
    loss, bboxes, scores, classes, valid_detections = validation(data['image'], data['label'])

    index = data['label_index']
    gt_box = generator.get_bbox(index)

    precision_50 = precision(gt_box, bboxes.numpy(),
                             valid_detections=valid_detections.numpy(), threshold=0.5)
    precision_75 = precision(gt_box, bboxes.numpy(),
                             valid_detections=valid_detections.numpy(), threshold=0.75)
    recall_50 = recall(gt_box, bboxes.numpy(), valid_detections=valid_detections.numpy(),
                       threshold=0.5)
    recall_75 = recall(gt_box, bboxes.numpy(), valid_detections=valid_detections.numpy(),
                       threshold=0.75)
    mAP_50 = mAP(gt_box, bboxes.numpy(), scores=scores.numpy(), valid_detections=valid_detections.numpy(),
                 threshold=0.5)
    mAP_75 = mAP(gt_box, bboxes.numpy(), scores=scores.numpy(), valid_detections=valid_detections.numpy(),
                 threshold=0.75)

    plt_image = plot_bounding_box(data['image'], bboxes, scores, valid_detections, ckpt.step, mode='train')

    with writer.as_default():
        tf.summary.scalar("lr", optimizer.lr, step=int(ckpt.step))
        tf.summary.scalar('loss', loss, step=int(ckpt.step))
        tf.summary.scalar('mean loss', loss / train_config['batch_size'], step=int(ckpt.step))
        tf.summary.scalar('precision@0.5', precision_50, step=int(ckpt.step))
        tf.summary.scalar('precision@0.75', precision_75, step=int(ckpt.step))
        tf.summary.scalar('recall@0.5', recall_50, step=int(ckpt.step))
        tf.summary.scalar('recall@0.75', recall_75, step=int(ckpt.step))
        tf.summary.scalar('mAP@0.5', mAP_50, step=int(ckpt.step))
        tf.summary.scalar('mAP@0.75', mAP_75, step=int(ckpt.step))
        tf.summary.image("Display bounding box", plt_image, step=int(ckpt.step))

    return loss


def train(dataset_train, dataset_val, train_generator, val_generator):
    # restore checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for data in dataset_train:
        train_loss = train_one_step(data['image'], data['label'])

        ckpt.step.assign_add(1)

        if 0 < train_loss / train_config['batch_size'] <= 2.0 or int(ckpt.step) >= train_config['total_steps']:
            print("Early stopping")
            print("Final training loss {:1.2f}".format(train_loss / train_config['batch_size']))
            model.save('./saved_model/yolov3_2.0')
            return

        if train_loss < 0:
            print("Error. Restart training from checkpoint again")
            train(dataset_train, dataset_val, train_generator, val_generator)

        if int(ckpt.step) % 1000 == 0:
            tf.print("Steps: ", int(ckpt.step))
            # validation ever 1000 epochs
            # Training set
            train_loss = metrics_logging(writer=train_summary_writer, data=data, generator=train_generator)

            # Validation set
            data_val = next(iter(dataset_val))
            val_loss = metrics_logging(writer=val_summary_writer, data=data_val, generator=val_generator)

            # Save checkpoint
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("training loss {:1.2f}".format(train_loss.numpy()))
            print("validation loss {:1.2f}".format(val_loss.numpy()))


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
    elif args.dataset == 'synthtext':
        train_generator = SynthTextGenerator(os.path.join(dataset_dir, 'gt.mat'),
                                             dataset_dir,
                                             mode='train',
                                             image_input_size=[args.image_size, args.image_size],
                                             anchors=yolo_anchors,
                                             anchor_masks=yolo_anchor_masks
                                             )
        val_generator = SynthTextGenerator(os.path.join(dataset_dir, 'gt.mat'),
                                           dataset_dir,
                                           mode='val',
                                           image_input_size=[args.image_size, args.image_size],
                                           anchors=yolo_anchors,
                                           anchor_masks=yolo_anchor_masks
                                           )
    else:
        train_generator = ReceiptGenerator(dataset_dir,
                                           image_input_size=[args.image_size, args.image_size],
                                           anchors=yolo_anchors,
                                           anchor_masks=yolo_anchor_masks
                                           )
        val_generator = ReceiptGenerator(dataset_dir,
                                         image_input_size=[args.image_size, args.image_size],
                                         anchors=yolo_anchors,
                                         anchor_masks=yolo_anchor_masks
                                         )

    train_generator.set_dataset_info()
    val_generator.set_dataset_info()

    dataset_train_generator = DetectorDataset(
        generator=train_generator,
        image_input_size=[args.image_size, args.image_size],
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )
    dataset_val_generator = DetectorDataset(
        generator=val_generator,
        image_input_size=[args.image_size, args.image_size],
        batch_size=args.batch_size,
        buffer_size=BUFFER_SIZE,
        prefetch_size=PREFETCH_SIZE
    )
    dataset_train = dataset_train_generator.create_dataset()
    dataset_val = dataset_val_generator.create_dataset()

    # setup runtime train config
    train_config['batch_size'] = args.batch_size
    train_config['dataset_size'] = len(train_generator.filenames)
    train_config['warmup_steps'] = WARMUP_EPOCHS * train_config['dataset_size'] // args.batch_size
    train_config['total_steps'] = TRAIN_EPOCHS * train_config['dataset_size'] // args.batch_size

    # train network
    train(dataset_train, dataset_val, train_generator, val_generator)

    # stop vm after training finished
    if args.s:
        os.system('sudo shutdown -h now')


def plot_bounding_box(imgs, labels, scores, valid_detections, steps, mode):
    img = imgs.numpy()[0]
    label = labels.numpy()[0][:valid_detections.numpy()[0]]
    score = scores.numpy()[0][:valid_detections.numpy()[0]]

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
    for i, (bbox, score) in enumerate(zip(label, score)):
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1) * img_w
        h = abs(y2 - y1) * img_h
        x, y = x1 * img_w, y1 * img_h

        rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                  edgecolor=c[i], facecolor='none')
        ax.annotate("{:1.2f}".format(score), (x, y - 4), color=c[i])
        ax.add_patch(rect)

    img_file = "./figure/{}/image_{}.png".format(mode, int(steps / 100 % 30))
    plt.savefig(img_file)
    plt.draw()
    plt.pause(0.01)

    image = plt.imread(img_file)
    image = np.expand_dims(image, axis=0)

    return image


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
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-i', '--image_size', type=int, default=IMAGE_SIZE, help='Reshape size of the image')
    parser.add_argument('-s', action='store_true', help='Shut down vm after training stop')
    args = parser.parse_args()

    main(args)

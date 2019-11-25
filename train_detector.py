from model.yolov3 import YoloV3
from coco_text.coco_text import COCO_Text
import tensorflow as tf
import numpy as np
import argparse
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset_choice = ['coco_text']
IMAGE_SIZE = 416
BATCH_SIZE = 32

IMAGES_IN_MEMORY = 320


def parse_label(label, original_dim, resize_dim):
    parsed_label = []
    for l in label:
        l = resize_label(l['bbox'], original_dim, resize_dim)
        l.append(1)
        parsed_label.append(l)

    return parsed_label


def resize_label(label, original_dim, resize_dim):
    img_w, img_h = original_dim
    target_w, target_h = resize_dim
    ratio_w = min(target_w / img_w, target_h / img_h)
    ratio_h = min(target_w / img_w, target_h / img_h)
    x, y, w, h = label

    # the label is normalized by the resized img size
    return [x * ratio_w / target_w, y * ratio_h / target_h, w * ratio_w / target_w, h * ratio_h / target_h]


def resize_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[0:new_h, 0:new_w, :] = resized_image

    return canvas


def load_dataset(ct, imgs_dir, imgs_meta):
    imgs_dir = ['%s/%s' % (imgs_dir, img_meta['file_name']) for img_meta in imgs_meta]
    labels = [load_label(ct, img_meta['id']) for img_meta in imgs_meta]

    resized_imgs = []
    resized_labels = []

    for i in range(len(imgs_meta)):
        img_w, img_h = imgs_meta[i]['width'], imgs_meta[i]['height']

        # resize image
        img = io.imread(imgs_dir[i])
        resized_img = resize_image(img, [IMAGE_SIZE, IMAGE_SIZE])

        # phase label
        resized_label = parse_label(labels[i], [img_w, img_h], [IMAGE_SIZE, IMAGE_SIZE])

        resized_imgs.append(resized_img)
        resized_labels.append(resized_label)

    return resized_imgs, resized_labels


def load_label(ct, img_id):
    return ct.loadAnns(ct.getAnnIds(imgIds=img_id))


def main(args):
    if args.dataset == 'coco_text':
        # set up dataset config
        ct = COCO_Text('./coco_text/cocotext.v2.json')
        data_type = 'train2014'
        imgs_dir = args.dir[0:-1] if args.dir[-1] == '/' else args.dir
        imgs_dir = '%s/%s' % (imgs_dir, data_type)

        # load images info
        imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible')])
        imgs_meta = ct.loadImgs(imgIds)

        img, label = load_dataset(ct, imgs_dir, imgs_meta[0:320])


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

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from dataset.coco_text.coco_text import COCO_Text


class COCOGenerator:
    def __init__(self, label_dir, dataset_dir, mode, batch_size, image_input_size, anchors, anchor_masks):
        self.ct = COCO_Text(label_dir)
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.batch_size = batch_size
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.img_metas = []
        self.faulty_img = json.load(open('dataset/coco_text/faulty_image.json', 'r'))['faulty_image']
        self.bboxes = np.zeros((1, 4))

    def set_img_metas(self):
        # set mode
        if self.mode == 'train':
            imgIds = self.ct.train
        elif self.mode == 'val':
            imgIds = self.ct.val
        else:
            imgIds = self.ct.test

        img_ids = self.ct.getImgIds(imgIds=imgIds, catIds=[('legibility', 'legible')])
        self.img_metas = self.ct.loadImgs(img_ids)

        if self.mode != 'train':
            self.img_metas = np.random.choice(self.img_metas, self.batch_size)

    def set_filename(self):
        # ['path_to_file', ...]
        for img_meta in self.img_metas:
            img_meta['file_name'] = '%s/%s' % (self.dataset_dir, img_meta['file_name'])

    def transform_targets_for_output(self, y_true, grid_size, anchor_idxs):
        # y_true: (boxes, (x, y, w, h, class, best_anchor))
        # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
        y_true_out = np.zeros((grid_size, grid_size, anchor_idxs.shape[0], 6))

        for i in range(y_true.shape[0]):
            anchor_eq = np.equal(
                anchor_idxs, y_true[i][5]
            )

            if np.any(anchor_eq):
                box = y_true[i][0:4]
                box_xy = y_true[i][0:2]

                anchor_idx = np.where(anchor_eq)
                grid_xy = box_xy // (1 / grid_size)
                grid_xy = grid_xy.astype(int)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                y_true_out[grid_xy[1]][grid_xy[0]][anchor_idx[0][0]] = [box[0], box[1], box[2], box[3], 1, y_true[i][4]]

        return y_true_out

    def transform_label(self, y_true):
        # y_train = [[x,y,w,h,c],...] shape=(n, 5)
        y_outs = []
        grid_size = math.ceil(self.image_input_size[0] / 32)

        anchor_area = self.anchors[..., 0] * self.anchors[..., 1]
        box_wh = y_true[..., 2:4]
        box_wh = np.tile(np.expand_dims(box_wh, -2), (1, 1, self.anchors.shape[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = np.minimum(box_wh[..., 0], self.anchors[..., 0]) * np.minimum(box_wh[..., 1],
                                                                                     self.anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = np.argmax(iou, axis=-1)
        anchor_idx = np.reshape(anchor_idx, (-1, 1))

        y_train = np.concatenate([y_true, anchor_idx], axis=-1)

        for anchor_idxs in self.anchor_masks:
            y_outs.append(self.transform_targets_for_output(y_train, grid_size, anchor_idxs))
            grid_size *= 2

        return y_outs

    def parse_label(self, fn, img_mate):
        # get bounding box
        label = fn(img_mate['id'])
        img_h, img_w = img_mate['height'], img_mate['width']
        label = np.asarray([l['bbox'] for l in label])

        # apply transformation to each label
        label = self.resize_label(label, [img_h, img_w])
        self.bboxes = np.append(self.bboxes, label, axis=0)

        # add class id to each bbox
        class_ids = np.ones((label.shape[0], 1))
        label = np.concatenate([label, class_ids], axis=-1)

        label = self.transform_label(label)

        return label

    def set_labels(self):
        fn = lambda x: self.ct.loadAnns(self.ct.getAnnIds(imgIds=x))
        labels = [self.parse_label(fn, img_mate) for img_mate in self.img_metas]

        self.labels = labels

    def transforming_center(self, label):
        for element in label:
            element[0] += element[2] / 2
            element[1] += element[3] / 2
        return label

    def resize_label(self, label, original_dim):
        # change top-left xy to center xy
        # [x, y, w, h] -> [center_x, center_y, w, h]
        label = self.transforming_center(label)

        # normalize label
        img_h, img_w = original_dim
        target_h, target_w = self.image_input_size
        ratio_w = min(target_w / img_w, target_h / img_h) / target_w
        ratio_h = min(target_w / img_w, target_h / img_h) / target_h

        index = label.shape[0]

        multiplier = np.asarray([[ratio_w, ratio_h, ratio_w, ratio_h] for _ in range(index)])

        return label * multiplier

    def clean_image(self):
        def filter_func(n):
            return n['file_name'] not in self.faulty_img

        self.img_metas = list(filter(filter_func, self.img_metas))

    def plt_img_dim_cluster(self):
        bboxes = self.bboxes
        w = bboxes[..., 2]
        h = bboxes[..., 3]

        plt.scatter(w, h, s=0.1)
        plt.show()
        plt.close()

    def set_dataset_info(self):
        self.set_img_metas()
        self.clean_image()
        self.set_filename()
        self.set_labels()

    def gen_next_pair(self):
        while True:
            index = np.random.randint(0, len(self.img_metas))

            img, label = self.img_metas[index]['file_name'], self.labels[index]
            scale_1_label, scale_2_label, scale_3_label = label[0], label[1], label[2]

            yield ({
                'image': img,
                'label': tuple([scale_1_label, scale_2_label, scale_3_label])
            })

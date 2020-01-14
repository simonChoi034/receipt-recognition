import math
import os

import imagesize
import numpy as np
import scipy.io as io


class SynthTextGenerator:
    def __init__(self, label_dir, dataset_dir, mode, image_input_size, anchors, anchor_masks):
        self.data = io.loadmat(label_dir)
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.filenames = None
        self.labels = None

    def set_filenames(self):
        filenames = self.data['imnames'][0]
        filenames = np.asarray(list(map(lambda x: os.path.join(self.dataset_dir, x[0]), filenames)))

        file_len = int(len(filenames) * 0.8)
        self.filenames = filenames[:file_len] if self.mode == 'train' else filenames[file_len:]

    def set_labels(self):
        labels = self.data['wordBB'][0]
        label_len = int(len(labels) * 0.8)
        labels = labels[:label_len] if self.mode == 'train' else labels[label_len:]
        self.labels = list(labels)

    def parse_bbox(self, label_tensor, filename):
        if np.ndim(label_tensor) == 2:
            label_tensor = np.expand_dims(label_tensor, -1)

        # find the dimension of the image
        width, height = imagesize.get(filename)

        bboxes = []
        for i in range(label_tensor.shape[-1]):
            x, y = label_tensor[..., i][0], label_tensor[..., i][1]
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            w, h = x_max - x_min, y_max - y_min
            x_cen, y_cen = x_min + w / 2, y_min + h / 2

            # skip bounding box if overflow
            if 0 <= x_cen <= width and 0 <= y_cen <= height:
                # append bounding box if not overflow
                bbox = [x_cen, y_cen, w, h]
                bboxes.append(bbox)

        bboxes = np.asarray(bboxes).reshape((-1, 4))

        bboxes = self.resize_label(bboxes, [height, width])
        class_id = np.ones((len(bboxes), 1))

        return np.concatenate((bboxes, class_id), axis=-1)

    def resize_label(self, label, original_dim):
        # normalize label
        img_h, img_w = original_dim
        target_h, target_w = self.image_input_size
        ratio_w = min(target_w / img_w, target_h / img_h) / target_w
        ratio_h = min(target_w / img_w, target_h / img_h) / target_h

        index = label.shape[0]

        multiplier = np.asarray([[ratio_w, ratio_h, ratio_w, ratio_h] for _ in range(index)])

        return label * multiplier

    def set_dataset_info(self):
        self.set_filenames()
        self.set_labels()

        del self.data

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

    def gen_next_pair(self):
        while True:
            index = np.random.randint(0, len(self.filenames))

            img, label = self.filenames[index], self.labels[index]

            label = self.parse_bbox(label, img)
            scale_1_label, scale_2_label, scale_3_label = self.transform_label(label)

            yield ({
                'image': img,
                'scale_1_label': scale_1_label,
                'scale_2_label': scale_2_label,
                'scale_3_label': scale_3_label
            })

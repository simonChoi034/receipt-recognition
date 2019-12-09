import math

import numpy as np
import tensorflow as tf

from coco_text.coco_text import COCO_Text

error_image = ['COCO_train2014_000000033127.jpg', 'COCO_train2014_000000166522.jpg', 'COCO_train2014_000000087509.jpg',
               'COCO_train2014_000000492325.jpg', 'COCO_train2014_000000264165.jpg', 'COCO_train2014_000000384907.jpg',
               'COCO_train2014_000000269858.jpg', 'COCO_train2014_000000389984.jpg', 'COCO_train2014_000000454000.jpg',
               'COCO_train2014_000000431115.jpg', 'COCO_train2014_000000223616.jpg', 'COCO_train2014_000000030349.jpg',
               'COCO_train2014_000000270925.jpg', 'COCO_train2014_000000225717.jpg', 'COCO_train2014_000000226585.jpg',
               'COCO_train2014_000000480482.jpg', 'COCO_train2014_000000564314.jpg', 'COCO_train2014_000000134918.jpg',
               'COCO_train2014_000000563447.jpg', 'COCO_train2014_000000107450.jpg', 'COCO_train2014_000000134071.jpg',
               'COCO_train2014_000000384910.jpg', 'COCO_train2014_000000397575.jpg', 'COCO_train2014_000000066642.jpg',
               'COCO_train2014_000000220898.jpg', 'COCO_train2014_000000484742.jpg', 'COCO_train2014_000000032405.jpg',
               'COCO_train2014_000000140092.jpg', 'COCO_train2014_000000061048.jpg', 'COCO_train2014_000000250239.jpg',
               'COCO_train2014_000000005294.jpg', 'COCO_train2014_000000576700.jpg', 'COCO_train2014_000000179405.jpg',
               'COCO_train2014_000000145288.jpg', 'COCO_train2014_000000011801.jpg', 'COCO_train2014_000000316867.jpg',
               'COCO_train2014_000000176483.jpg', 'COCO_train2014_000000384693.jpg', 'COCO_train2014_000000257178.jpg',
               'COCO_train2014_000000381270.jpg', 'COCO_train2014_000000518025.jpg', 'COCO_train2014_000000015286.jpg',
               'COCO_train2014_000000449901.jpg', 'COCO_train2014_000000001350.jpg', 'COCO_train2014_000000140627.jpg',
               'COCO_train2014_000000369966.jpg', 'COCO_train2014_000000033352.jpg', 'COCO_train2014_000000015236.jpg',
               'COCO_train2014_000000057978.jpg', 'COCO_train2014_000000457741.jpg', 'COCO_train2014_000000207339.jpg',
               'COCO_train2014_000000336668.jpg', 'COCO_train2014_000000509358.jpg', 'COCO_train2014_000000000821.jpg',
               'COCO_train2014_000000496444.jpg', 'COCO_train2014_000000363331.jpg', 'COCO_train2014_000000113929.jpg',
               'COCO_train2014_000000155954.jpg', 'COCO_train2014_000000525513.jpg', 'COCO_train2014_000000204792.jpg',
               'COCO_train2014_000000342921.jpg', 'COCO_train2014_000000118895.jpg', 'COCO_train2014_000000107962.jpg',
               'COCO_train2014_000000280731.jpg', 'COCO_train2014_000000349069.jpg', 'COCO_train2014_000000394547.jpg',
               'COCO_train2014_000000259284.jpg', 'COCO_train2014_000000343009.jpg', 'COCO_train2014_000000321897.jpg',
               'COCO_train2014_000000233263.jpg', 'COCO_train2014_000000003293.jpg', 'COCO_train2014_000000176397.jpg',
               'COCO_train2014_000000150354.jpg', 'COCO_train2014_000000555583.jpg', 'COCO_train2014_000000293833.jpg',
               'COCO_train2014_000000173610.jpg', 'COCO_train2014_000000421613.jpg', 'COCO_train2014_000000095753.jpg',
               'COCO_train2014_000000520479.jpg', 'COCO_train2014_000000470933.jpg', 'COCO_train2014_000000018702.jpg',
               'COCO_train2014_000000140623.jpg', 'COCO_train2014_000000325387.jpg', 'COCO_train2014_000000104124.jpg',
               'COCO_train2014_000000353952.jpg', 'COCO_train2014_000000220770.jpg', 'COCO_train2014_000000249711.jpg',
               'COCO_train2014_000000156878.jpg', 'COCO_train2014_000000085407.jpg', 'COCO_train2014_000000503640.jpg',
               'COCO_train2014_000000313608.jpg', 'COCO_train2014_000000260962.jpg', 'COCO_train2014_000000390663.jpg',
               'COCO_train2014_000000347111.jpg', 'COCO_train2014_000000075052.jpg', 'COCO_train2014_000000563376.jpg',
               'COCO_train2014_000000072098.jpg', 'COCO_train2014_000000228474.jpg', 'COCO_train2014_000000221691.jpg',
               'COCO_train2014_000000029275.jpg', 'COCO_train2014_000000579138.jpg', 'COCO_train2014_000000205486.jpg',
               'COCO_train2014_000000507794.jpg', 'COCO_train2014_000000416869.jpg', 'COCO_train2014_000000077709.jpg',
               'COCO_train2014_000000426558.jpg', 'COCO_train2014_000000058517.jpg', 'COCO_train2014_000000312288.jpg',
               'COCO_train2014_000000035880.jpg', 'COCO_train2014_000000443909.jpg', 'COCO_train2014_000000208206.jpg',
               'COCO_train2014_000000451074.jpg', 'COCO_train2014_000000012345.jpg', 'COCO_train2014_000000571415.jpg',
               'COCO_train2014_000000436984.jpg', 'COCO_train2014_000000060060.jpg', 'COCO_train2014_000000434765.jpg',
               'COCO_train2014_000000186888.jpg', 'COCO_train2014_000000505962.jpg', 'COCO_train2014_000000243205.jpg',
               'COCO_train2014_000000377837.jpg', 'COCO_train2014_000000210847.jpg', 'COCO_train2014_000000470442.jpg',
               'COCO_train2014_000000131366.jpg', 'COCO_train2014_000000361516.jpg', 'COCO_train2014_000000434837.jpg',
               'COCO_train2014_000000040428.jpg', 'COCO_train2014_000000540378.jpg', 'COCO_train2014_000000577265.jpg',
               'COCO_train2014_000000249835.jpg', 'COCO_train2014_000000006432.jpg', 'COCO_train2014_000000296884.jpg',
               'COCO_train2014_000000341892.jpg', 'COCO_train2014_000000210175.jpg', 'COCO_train2014_000000445845.jpg',
               'COCO_train2014_000000416372.jpg', 'COCO_train2014_000000549879.jpg', 'COCO_train2014_000000578250.jpg',
               'COCO_train2014_000000217341.jpg', 'COCO_train2014_000000010125.jpg', 'COCO_train2014_000000008794.jpg',
               'COCO_train2014_000000123539.jpg', 'COCO_train2014_000000025404.jpg', 'COCO_train2014_000000080906.jpg',
               'COCO_train2014_000000517899.jpg', 'COCO_train2014_000000155083.jpg', 'COCO_train2014_000000579239.jpg',
               'COCO_train2014_000000476888.jpg', 'COCO_train2014_000000081003.jpg', 'COCO_train2014_000000263002.jpg',
               'COCO_train2014_000000126531.jpg', 'COCO_train2014_000000000086.jpg', 'COCO_train2014_000000571503.jpg',
               'COCO_train2014_000000400107.jpg', 'COCO_train2014_000000268036.jpg', 'COCO_train2014_000000053756.jpg',
               'COCO_train2014_000000410498.jpg', 'COCO_train2014_000000561842.jpg', 'COCO_train2014_000000124694.jpg',
               'COCO_train2014_000000211867.jpg', 'COCO_train2014_000000034861.jpg', 'COCO_train2014_000000577207.jpg',
               'COCO_train2014_000000287422.jpg', 'COCO_train2014_000000006379.jpg', 'COCO_train2014_000000358281.jpg',
               'COCO_train2014_000000027412.jpg', 'COCO_train2014_000000406011.jpg', 'COCO_train2014_000000518951.jpg',
               'COCO_train2014_000000084582.jpg', 'COCO_train2014_000000575029.jpg', 'COCO_train2014_000000064270.jpg']


class COCOGenerator:
    def __init__(self, label_dir, dataset_dir, image_input_size, anchors, anchor_masks):
        self.ct = COCO_Text(label_dir)
        self.dataset_dir = dataset_dir
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.img_metas = []

    def set_img_metas(self):
        img_ids = self.ct.getImgIds(imgIds=self.ct.train, catIds=[('legibility', 'legible')])
        self.img_metas = self.ct.loadImgs(img_ids)

    def set_filename(self):
        # ['path_to_file', ...]
        for img_meta in self.img_metas:
            img_meta['file_name'] = '%s/%s' % (self.dataset_dir, img_meta['file_name'])

    def transform_targets_for_output(self, y_true, grid_size, anchor_idxs):
        # y_true: (boxes, (x1, y1, x2, y2, class, best_anchor))
        # y_true_out: (grid, grid, anchors, [x1, y1, x2, y2, obj, class])
        y_true_out = np.zeros((grid_size, grid_size, anchor_idxs.shape[0], 6))

        for i in range(y_true.shape[0]):
            if y_true[i][2] == 0:
                continue
            anchor_eq = np.equal(
                anchor_idxs, y_true[i][5]
            )

            if np.any(anchor_eq):
                box = y_true[i][0:4]
                box_xy = (y_true[i][0:2] + y_true[i][2:4]) / 2

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
        box_wh = y_true[..., 2:4] - y_true[..., 0:2]
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
        label = [l['bbox'] for l in label]
        # apply transformation to each label
        label = self.resize_label(np.array(label), [img_h, img_w], self.image_input_size)
        # add class id to each bbox
        mapfun = lambda x: np.append(x, 1)
        label = np.apply_along_axis(mapfun, 1, label)

        label = self.transform_label(label)

        return label

    def set_labels(self):
        fn = lambda x: self.ct.loadAnns(self.ct.getAnnIds(imgIds=x))
        labels = [self.parse_label(fn, img_mate) for img_mate in self.img_metas]

        self.labels = labels

    def transforming_wh(self, label):
        for element in label:
            element[2] += element[0]
            element[3] += element[1]
        return label

    def resize_label(self, label, original_dim, resize_dim):
        # change top-left xy to center xy
        # [x, y, w, h] -> [x.min, y.min, x.max, y.max]
        label = self.transforming_wh(label)

        # normalize label
        img_h, img_w = original_dim
        target_h, target_w = resize_dim
        ratio_w = min(target_w / img_w, target_h / img_h) / target_w
        ratio_h = min(target_w / img_w, target_h / img_h) / target_h

        index = label.shape[0]

        multiplier = [[ratio_w, ratio_h, ratio_w, ratio_h] for _ in range(index)]
        multiplier = np.array(multiplier)

        return label * multiplier

    def clean_image(self):
        def filter_fun(n):
            return n['file_name'] not in error_image

        self.img_metas = list(filter(filter_fun, self.img_metas))

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
                'scale_1_label': scale_1_label,
                'scale_2_label': scale_2_label,
                'scale_3_label': scale_3_label
            })


class Dataset:
    def __init__(self, generator, image_input_size, batch_size, buffer_size, prefetch_size):
        self.generator = generator
        self.image_input_size = image_input_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size

    def resize_image(self, img, inp_dim):
        img = tf.image.resize(img, inp_dim, preserve_aspect_ratio=True)
        img = tf.image.pad_to_bounding_box(img, 0, 0, inp_dim[0], inp_dim[1])
        return img / 127.5 - 1  # normalize to [-1, 1]

    def read_and_resize_image(self, element):
        img = tf.io.read_file(element['image'])
        img = tf.image.decode_jpeg(img)
        img.set_shape([None, None, 3])
        scale_1_label, scale_2_label, scale_3_label = element['scale_1_label'], element['scale_2_label'], element[
            'scale_3_label']

        # resize and pad image to required input size
        img = self.resize_image(img, self.image_input_size)
        # format label

        element['image'] = img
        element['label'] = tuple([scale_1_label, scale_2_label, scale_3_label])

        return element

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generator.gen_next_pair,
            output_types={
                'image': tf.string,
                'scale_1_label': tf.float32,
                'scale_2_label': tf.float32,
                'scale_3_label': tf.float32
            }
        )
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(map_func=self.read_and_resize_image)
        dataset = dataset.filter(lambda x: x['image'].shape[2] == 3)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        return dataset

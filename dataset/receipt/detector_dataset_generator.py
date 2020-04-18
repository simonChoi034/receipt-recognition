import json
import math
import os
import re

import imagesize
import matplotlib.pyplot as plt
import numpy as np
import chars2vec

class_ids = {
    "MerchantName": 1,
    "MerchantAddress": 2,
    "TransactionDate": 3,
    "Total": 4
}


class ReceiptGenerator:
    def __init__(self, dataset_dir, image_input_size, anchors, anchor_masks):
        self.filenames = sorted(
            [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if re.match(r'X[0-9]+\.jpg', f)])
        self.label_files = sorted(
            [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if re.match(r'X[0-9]+\.txt', f)])
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks

    def read_label_file(self, file):
        image_file = file.replace('txt', 'jpg')
        width, height = imagesize.get(image_file)

        bboxes = []
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                line = np.asarray(line[:8]).astype(float)
                line = line.reshape((4, 2))

                x, y = line[..., 0], line[..., 1]
                x_min, x_max = min(x), max(x)
                y_min, y_max = min(y), max(y)
                w, h = x_max - x_min, y_max - y_min
                x_cen, y_cen = x_min + w / 2, y_min + h / 2

                bbox = [x_cen, y_cen, w, h]
                bboxes.append(bbox)

        bboxes = self.resize_label(np.asarray(bboxes), [height, width])
        class_id = np.ones((len(bboxes), 1))

        return np.concatenate((bboxes, class_id), axis=-1)

    def resize_label(self, label, original_dim):
        # change top-left xy to center xy
        # [x, y, w, h] -> [center_x, center_y, w, h]

        # normalize label
        img_h, img_w = original_dim
        target_h, target_w = self.image_input_size
        ratio_w = min(target_w / img_w, target_h / img_h) / target_w
        ratio_h = min(target_w / img_w, target_h / img_h) / target_h

        index = label.shape[0]

        multiplier = np.asarray([[ratio_w, ratio_h, ratio_w, ratio_h] for _ in range(index)])

        return label * multiplier

    def set_labels(self):
        bboxes = np.asarray([self.read_label_file(file) for file in self.label_files])
        labels = np.asarray([self.transform_label(label) for label in bboxes])
        self.bboxes = bboxes
        self.labels = labels

    def set_dataset_info(self):
        self.set_labels()

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

    def plt_img_dim_cluster(self):
        bboxes = self.bboxes[0]
        for l in self.bboxes:
            bboxes = np.concatenate((bboxes, l), axis=0)
        print(bboxes.shape)
        w = bboxes[..., 2]
        h = bboxes[..., 3]

        plt.scatter(w, h, s=0.1)
        plt.show()
        plt.close()

    def gen_next_pair(self):
        while True:
            index = np.random.randint(0, len(self.filenames))

            img, label = self.filenames[index], self.labels[index]
            scale_1_label, scale_2_label, scale_3_label = label[0], label[1], label[2]

            yield ({
                'image': img,
                'scale_1_label': scale_1_label,
                'scale_2_label': scale_2_label,
                'scale_3_label': scale_3_label
            })


class ReceiptClassifyGenerator:
    def __init__(self, dataset_dir, vocab_size, word_size, char_size, mode):
        self.vocab_size = vocab_size  # 128 -> ascii number
        self.word_size = word_size
        self.char_size = char_size
        self.mode = mode
        self.c2v_model = chars2vec.load_model('eng_{}'.format(char_size))
        # dir = <dir>/{train|val}/<filename>.json
        self.filenames = sorted(
            [os.path.join(dataset_dir, mode, f) for f in os.listdir(os.path.join(dataset_dir, mode)) if
             re.match(r'.*\.json', f)])
        self.data = [self.read_file(file) for file in self.filenames]
        self.document_lists = []
        self.labels = []

    def read_file(self, file):
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            return data['analyzeResult']

    def crop_or_pad_zero(self, array, num):
        arr_len = len(array)
        if arr_len < num:
            return np.pad(array, (0, num - arr_len), 'constant')
        else:
            return array[:num]

    def transform_ascii(self, string):
        return [ord(c) for c in string if 0 <= ord(c) < 128]

    def transform_data(self, array):
        word_list = [str(ele['text']) for ele in array]
        class_inx = [int(ele['class']) for ele in array]

        # encode words with pre-trained char2vec model
        word_list = self.c2v_model.vectorize_words(word_list)

        # pad class ids
        class_inx = self.crop_or_pad_zero(class_inx, self.word_size)

        # pad encoded words
        word_len = len(word_list)
        zeros = np.zeros((self.word_size - word_len, self.char_size))
        word_list = np.concatenate([word_list, zeros], axis=0)

        return word_list, class_inx

    def pad_class_id(self):
        for document in self.data:
            # initial all words class id to 0
            for page in document['readResults']:
                for line in page['lines']:
                    for word in line['words']:
                        word['class'] = 0

            # set class id for each words
            document_results = document['documentResults']
            for document_result in document_results:
                class_fields = document_result['fields']
                for class_key, class_id in class_ids.items():
                    if class_key in class_fields:
                        for word_element in class_fields[class_key]['elements']:
                            word_element = word_element.split('/')
                            page_idx = int(word_element[2])
                            line_idx = int(word_element[4])
                            word_idx = int(word_element[6])
                            document['readResults'][page_idx]['lines'][line_idx]['words'][word_idx]['class'] = class_id

    def create_word_array(self):
        for document in self.data:
            # create array of words and class id
            document_words = []
            for page in document['readResults']:
                for line in page['lines']:
                    for word in line['words']:
                        # word = { text, class_id }
                        document_words.append(word)

            # encode words and label
            document_words, label = self.transform_data(document_words)

            self.document_lists.append(document_words)
            self.labels.append(label)

        # create np array
        self.document_lists = np.asarray(self.document_lists)
        self.labels = np.asarray(self.labels)

    def set_dataset_info(self):
        self.pad_class_id()
        self.create_word_array()

    def gen_next_pair(self):
        while True:
            index = np.random.randint(0, len(self.filenames))

            word_list, label = self.document_lists[index], self.labels[index]

            yield ({
                'word_list': word_list,
                'label': label
            })


class GridReceiptClassifyGenerator:
    def __init__(self, dataset_dir, vocab_size, word_size, char_size, grid_size, mode):
        self.vocab_size = vocab_size  # 128 -> ascii number
        self.word_size = word_size
        self.char_size = char_size
        self.grid_size = grid_size
        self.c2v_model = chars2vec.load_model('eng_{}'.format(char_size))
        self.mode = mode
        # dir = <dir>/{train|val}/<filename>.json
        self.filenames = sorted(
            [os.path.join(dataset_dir, mode, f) for f in os.listdir(os.path.join(dataset_dir, mode)) if
             re.match(r'.*\.json', f)])
        self.data = [self.read_file(file) for file in self.filenames]
        self.grids = []
        self.labels = []

    def read_file(self, file):
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            return data['analyzeResult']

    def pad_class_id(self):
        for document in self.data:
            # initial all words class id to 0
            for page in document['readResults']:
                for line in page['lines']:
                    for word in line['words']:
                        word['class'] = 0

            # set class id for words
            document_results = document['documentResults']
            for document_result in document_results:
                class_fields = document_result['fields']
                for class_key, class_id in class_ids.items():
                    if class_key in class_fields:
                        for word_element in class_fields[class_key]['elements']:
                            word_element = word_element.split('/')
                            page_idx = int(word_element[2])
                            line_idx = int(word_element[4])
                            word_idx = int(word_element[6])
                            document['readResults'][page_idx]['lines'][line_idx]['words'][word_idx]['class'] = class_id

    def create_grid(self, document):
        input_grid = np.zeros((self.grid_size[0], self.grid_size[1], self.char_size))
        label_grid = np.zeros((self.grid_size[0], self.grid_size[1]))
        for page in document['readResults']:
            width, height = page['width'], page['height']
            for line in page['lines']:
                for word in line['words']:
                    # calculate grid index
                    x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                    x_min, x_max = min([x1, x2, x3, x4]), max([x1, x2, x3, x4])
                    y_min, y_max = min([y1, y2, y3, y4]), max([y1, y2, y3, y4])
                    w, h = x_max - x_min, y_max - y_min
                    x_cen, y_cen = x_min + w / 2, y_min + h / 2
                    x_cen, y_cen = x_cen / width, y_cen / height

                    column_index = int(x_cen * self.grid_size[1])
                    row_index = int(y_cen * self.grid_size[0])

                    # encode text and class_id
                    class_id = int(word['class'])
                    text = [str(word['text'])]
                    encoded_text = self.c2v_model.vectorize_words(text)

                    input_grid[row_index][column_index] = encoded_text[0]
                    label_grid[row_index][column_index] = class_id

        return input_grid, label_grid

    def crop_or_pad_zero(self, array, num):
        arr_len = len(array)
        if arr_len < num:
            return np.pad(array, (0, num - arr_len), 'constant')
        else:
            return array[:num]

    def transform_ascii(self, string):
        return [ord(c) for c in string if 0 <= ord(c) < 128]

    def transform_data(self):
        for document in self.data:
            # get input grid for each receipt
            input_grid, label_grid = self.create_grid(document)
            self.grids.append(input_grid)
            self.labels.append(label_grid)

        self.grids = np.asarray(self.grids)
        self.labels = np.asarray(self.labels)

    def set_dataset_info(self):
        self.pad_class_id()
        self.transform_data()

    def gen_next_pair(self):
        while True:
            index = np.random.randint(0, len(self.filenames))

            grid, label = self.grids[index], self.labels[index]

            yield ({
                'word_list': grid,
                'label': label
            })

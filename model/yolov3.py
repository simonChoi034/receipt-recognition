import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

from .darknet import Darknet53
from .layers import MyConv2D, ResidualBlock, Concatenate, UpSampling2D


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


class YoloV3(tf.keras.Model):
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    def __init__(self, num_class, name='yolov3', **kwargs):
        super(YoloV3, self).__init__(name=name, **kwargs)
        self.num_class = num_class
        self.darknet = Darknet53()
        self.upsampling2d = UpSampling2D()
        self.concat = Concatenate()
        self.yolo_iou_threshold = 0.5
        self.yolo_score_threshold = 0.5

    def yolo_output(self, input, anchors):
        x = MyConv2D(filters=anchors * (self.num_class + 5), kernel_size=1, apply_batchnorm=False)(input)
        x = tf.reshape(
            x,
            (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, self.num_class + 5)
        )
        return x

    ### reference code from https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
    def yolo_boxes(self, pred, anchors):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, self.num_class), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
                 tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box

    def output_bbox(self, input):
        b, c, t = [], [], []

        for o in input:
            b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

        bbox = tf.concat(b, axis=1)
        confidence = tf.concat(c, axis=1)
        class_probs = tf.concat(t, axis=1)

        scores = confidence * class_probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=self.yolo_iou_threshold,
            score_threshold=self.yolo_score_threshold
        )

        return boxes, scores, classes, valid_detections

    def yolo_loss(self, anchors, ignore_thresh=0.5):
        def yolo_loss(y_true, y_pred):
            # 1. transform all pred outputs
            # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
            pred_box, pred_obj, pred_class, pred_xywh = self.yolo_boxes(
                y_pred, anchors)
            pred_xy = pred_xywh[..., 0:2]
            pred_wh = pred_xywh[..., 2:4]

            # 2. transform all true outputs
            # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
            true_box, true_obj, true_class_idx = tf.split(
                y_true, (4, 1, 1), axis=-1)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]

            # give higher weights to small boxes
            box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

            # 3. inverting the pred box equations
            grid_size = tf.shape(y_true)[1]
            grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
            true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                      tf.cast(grid, tf.float32)
            true_wh = tf.math.log(true_wh / anchors)
            true_wh = tf.where(tf.math.is_inf(true_wh),
                               tf.zeros_like(true_wh), true_wh)

            # 4. calculate all masks
            obj_mask = tf.squeeze(true_obj, -1)
            # ignore false positive when iou is over threshold
            true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
            best_iou = tf.reduce_max(broadcast_iou(
                pred_box, true_box_flat), axis=-1)
            ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

            # 5. calculate all losses
            xy_loss = obj_mask * box_loss_scale * \
                      tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * box_loss_scale * \
                      tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            obj_loss = binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + \
                       (1 - obj_mask) * ignore_mask * obj_loss
            # TODO: use binary_crossentropy instead
            class_loss = obj_mask * sparse_categorical_crossentropy(
                true_class_idx, pred_class)

            # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
            class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

            return xy_loss + wh_loss + obj_loss + class_loss

        return yolo_loss

    ### reference code from https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py

    def call(self, inputs, training=False, **kwargs):
        anchors = self.yolo_anchors
        masks = self.yolo_anchor_masks

        scale3, scale2, scale1 = self.darknet(inputs)

        # scale 1 output
        scale_1_detector = self.yolo_output(scale1, len(masks[0]))

        # scale 2 output
        scale_2_detector = MyConv2D(filters=256, kernel_size=1)(scale1)
        scale_2_detector = self.upsampling2d(scale_2_detector)
        scale_2_detector = self.concat([scale2, scale_2_detector])
        for i in range(0, 3):
            scale_2_detector = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])(scale_2_detector)
        # FPN shortcut
        scale_3_shortcut = scale_2_detector
        # scale 2 output
        scale_2_detector = self.yolo_output(scale_2_detector, len(masks[1]))

        # scale 3 output
        scale_3_detector = MyConv2D(filters=128, kernel_size=1)(scale_3_shortcut)
        scale_3_detector = self.upsampling2d(scale_3_detector)
        scale_3_detector = self.concat([scale3, scale_3_detector])
        for i in range(0, 3):
            scale_3_detector = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])(scale_3_detector)
        # scale 3 output
        scale_3_detector = self.yolo_output(scale_3_detector, len(masks[2]))

        if training:
            return scale_1_detector, scale_2_detector, scale_3_detector

        # bounding box prediction
        boxes_1 = self.yolo_boxes(scale_1_detector, anchors[masks[0]])
        boxes_2 = self.yolo_boxes(scale_2_detector, anchors[masks[1]])
        boxes_3 = self.yolo_boxes(scale_3_detector, anchors[masks[2]])

        output = self.output_bbox((boxes_1[:3], boxes_2[:3], boxes_3[:3]))

        return output

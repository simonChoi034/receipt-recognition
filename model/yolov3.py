import tensorflow as tf
from .layers import MyConv2D, ResidualBlock, Concatenate, UpSampling2D, Lambda
from .darknet import Darknet53


class YoloV3(tf.keras.Model):
    def __init__(self, num_class, num_anchors=3, is_training=False, name='yolov3', **kwargs):
        super(YoloV3, self).__init__(name=name, **kwargs)
        self.num_anchors = num_anchors
        self.num_class = num_class
        self.output_filters = num_anchors * (num_class + 5)
        self.is_training = is_training
        self.darknet = Darknet53()
        self.upsampling2d = UpSampling2D()
        self.concat = Concatenate()
        self.yolo_iou_threshold = 0.5
        self.yolo_score_threshold = 0.5

    def yolo_output(self, input):
        x = MyConv2D(filters=self.output_filters, kernel_size=1, apply_batchnorm=False)(input)
        x = tf.reshape(
            x,
            (-1, tf.shape(x)[1], tf.shape(x)[2], self.num_anchors, self.num_class + 5)
        )
        return x

    ### reference code from https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
    def yolo_boxes(self, pred):
        grid_size = tf.shape(pred)[1]
        box_xy, box_wh, prob_object, prob_class = tf.split(pred, (2, 2, 1, self.num_class), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        prob_object = tf.sigmoid(prob_object)
        prob_class = tf.sigmoid(prob_class)
        pred_box = tf.concat([box_xy, box_wh], axis=-1)

        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        box_xy = (box_xy + grid) / grid_size
        box_wh = tf.exp(box_wh) * self.num_anchors

        box_xy_1 = box_xy - box_wh / 2
        box_xy_2 = box_xy + box_wh / 2
        bounding_box = tf.concat([box_xy_1, box_xy_2], axis=-1)

        return bounding_box, prob_object, prob_class, pred_box

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
    ###



    def call(self, inputs, **kwargs):
        scale3, scale2, scale1 = self.darknet(inputs)

        # 13x13 output
        scale_1_detector = self.yolo_output(scale1)

        # 26x26 output
        scale_2_detector = MyConv2D(filters=256, kernel_size=1)(scale1)
        scale_2_detector = self.upsampling2d(scale_2_detector)
        scale_2_detector = self.concat([scale2, scale_2_detector])
        for i in range(0, 3):
            scale_2_detector = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])(scale_2_detector)
        # FPN shortcut
        scale_3_shortcut = scale_2_detector
        # scale 2 output
        scale_2_detector = self.yolo_output(scale_2_detector)

        # 52x52 output
        scale_3_detector = MyConv2D(filters=128, kernel_size=1)(scale_3_shortcut)
        scale_3_detector = self.upsampling2d(scale_3_detector)
        scale_3_detector = self.concat([scale3, scale_3_detector])
        for i in range(0, 3):
            scale_3_detector = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])(scale_3_detector)
        # scale 3 output
        scale_3_detector = self.yolo_output(scale_3_detector)

        if self.is_training:
            return scale_3_detector, scale_2_detector, scale_1_detector

        # bounding box prediction
        boxes_1 = self.yolo_boxes(scale_1_detector)
        boxes_2 = self.yolo_boxes(scale_2_detector)
        boxes_3 = self.yolo_boxes(scale_3_detector)

        output = self.output_bbox((boxes_1[:3], boxes_2[:3], boxes_3[:3]))

        return output
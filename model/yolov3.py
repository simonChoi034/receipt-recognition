import tensorflow as tf
from .layers import MyConv2D, ResidualBlock, Concatenate, UpSampling2D
from .darknet import Darknet53

class YoloV3(tf.keras.Model):
    def __init__(self, num_class, num_anchors=3, name='yolov3', **kwargs):
        super(YoloV3, self).__init__(name=name, **kwargs)
        self.darknet = Darknet53()
        self.output_filters = num_anchors * (num_class+5)
        self.upsampling2d = UpSampling2D()
        self.concat = Concatenate()


    def call(self, inputs, **kwargs):
        scale3, scale2, scale1 = self.darknet(inputs)

        # 13x13 output
        scale_1_detector = MyConv2D(filters=self.output_filters, kernel_size=1)(scale1)

        # 26x26 output
        scale_2_detector = MyConv2D(filters=256, kernel_size=1)(scale1)
        scale_2_detector = self.upsampling2d(scale_2_detector)
        scale_2_detector = self.concat([scale2, scale_2_detector])
        for i in range(0, 3):
            scale_2_detector = ResidualBlock(filters=[256, 512], kernel_size=[1, 3])(scale_2_detector)
        # FPN shortcut
        scale_3_shortcut = scale_2_detector
        # scale 2 output
        scale_2_detector = MyConv2D(filters=self.output_filters, kernel_size=1)(scale_2_detector)

        #52x52 output
        scale_3_detector = MyConv2D(filters=128, kernel_size=1)(scale_3_shortcut)
        scale_3_detector = self.upsampling2d(scale_3_detector)
        scale_3_detector = self.concat([scale3, scale_3_detector])
        for i in range(0, 3):
            scale_3_detector = ResidualBlock(filters=[128, 256], kernel_size=[1, 3])(scale_3_detector)
        # scale 3 output
        scale_3_detector = MyConv2D(filters=self.output_filters, kernel_size=1)(scale_3_detector)

        return scale_3_detector, scale_2_detector, scale_1_detector
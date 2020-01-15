import tensorflow as tf


class DetectorDataset:
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
        element['image'] = img
        # format label
        element['label'] = tuple([scale_1_label, scale_2_label, scale_3_label])

        return element

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generator.gen_next_pair,
            output_types={
                'image': tf.string,
                'scale_1_label': tf.float32,
                'scale_2_label': tf.float32,
                'scale_3_label': tf.float32,
                'label_index': tf.int32
            }
        )
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(map_func=self.read_and_resize_image)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        return dataset


class ClassifierDataset:
    def __init__(self, generator):
        self.generator = generator

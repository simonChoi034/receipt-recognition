import numpy as np
import cv2

char_dict = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"
params = {'SEQ_LENGTH': 47,
          'INPUT_SIZE': [200, 32],
          'NUM_CLASSES': len(char_dict)}


def decode_to_text(char_dict, decoded_out):
    return ''.join([char_dict[i] for i in decoded_out])


def sparse_tuple_from(sequences):
    indices = []
    values = []
    
    for n, m in enumerate(sequences):
        indices.extend(zip([n] * len(m), range(len(m))))
        values.extend(m)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    dense_shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, dense_shape


def preprocess_input_image(image, height=params['INPUT_SIZE'][1], width=params['INPUT_SIZE'][0]):

    scale_rate = height / image.shape[0]
    tmp_new_width = int(scale_rate * image.shape[1])
    new_width = width if tmp_new_width > width else tmp_new_width
    image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)

    r, c = np.shape(image)
    if c > width:
        ratio = float(width) / c
        image = cv2.resize(image, (width, int(32 * ratio)))
    else:
        width_pad = width - image.shape[1]
        image = np.pad(image, pad_width=[(0, 0), (0, width_pad)], mode='constant', constant_values=0)

    image = image[:, :, np.newaxis]

    return image


def data_generator(batches=1,
                   batch_size=2,
                   epochs=1,
                   char_dict=char_dict,
                   data_path='F:/mjsynth/mnt/ramdisk/max/90kDICT32px/', #dataset directory
                   dataset='train'  
                   # dataset
                   # training -> 'train', 
                   # testing -> 'test' ,or 
                   # validation -> 'val'
                   ):

    x_batch = []
    y_batch = []
    for _ in range(epochs):
        with open(data_path + 'annotation_{}.txt'.format(dataset)) as fp:
            for _ in range(batches * batch_size):
                image_path = fp.readline().replace('\n', '').split(' ')[0]

                # get x (image data)
                image = cv2.imread(data_path + image_path.replace('./', ''), 0)
                if image is None:
                    continue
                x = preprocess_input_image(image)

                # get y (true result)
                y = image_path.split('_')[1]
                y = [char_dict.index(i) if i in char_dict else len(char_dict)-1 for i in y]
                y = y  

                x_batch.append(x)
                y_batch.append(y)

                if len(y_batch) == batch_size:
                    yield np.array(x_batch).astype(np.float32), np.array(y_batch)
                    x_batch = []
                    y_batch = []


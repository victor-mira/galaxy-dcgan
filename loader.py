import os

import numpy as np
import tensorflow as tf
import cv2

class Loader:
    def __init__(self, filepath, buffer_size, batch_size):
        self.filepath = filepath
        img_array = []
        for filename in os.listdir(filepath):
            # print(filename)
            try:
                img = cv2.imread(filepath + "/" + filename)  # 返回numpy.ndarray
                img = cv2.resize(img, (96, 96))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array.append(img)

            except Exception as e:
                print(str(e))

        self.train_data = np.array(img_array).astype('float32')
        self.train_data = self.train_data/255

        self.train_ds = tf.data.Dataset.from_tensor_slices(self.train_data).shuffle(buffer_size).batch(batch_size)

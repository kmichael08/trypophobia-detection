import h5py
import numpy as np
from keras.utils import np_utils


def get_images(path):
    input_datafile = h5py.File(path)

    train_norm_images = np.array(input_datafile['train_norm'])
    train_trypo_images = np.array(input_datafile['train_trypo'])
    valid_norm_images = np.array(input_datafile['valid_norm'])
    valid_trypo_images = np.array(input_datafile['valid_trypo'])
    return train_norm_images, train_trypo_images, valid_norm_images, valid_trypo_images

def process_images(image, image_size=224):
    return np.divide(image, image_size)

def join_data(norm, trypo):
    X = np.concatenate([norm, trypo])
    Y1 = np.zeros(len(norm))
    Y2 = np.ones(len(trypo))
    Y = np.concatenate([Y1, Y2])
    Y = np_utils.to_categorical(Y)
    return X, Y

def transform_dataset(train_norm_images, train_trypo_images, valid_norm_images, valid_trypo_images):
    train_norm_images = process_images(train_norm_images)
    train_trypo_images = process_images(train_trypo_images)
    valid_norm_images = process_images(valid_norm_images)
    valid_trypo_images = process_images(valid_trypo_images)

    X, Y = join_data(train_norm_images, train_trypo_images)
    X_val, Y_val = join_data(valid_norm_images, valid_trypo_images)

    return X, Y, X_val, Y_val

if __name__ == "__main__":
    print(get_images('/media/michal/USB DISK/tryp.hdf5'))
    pass
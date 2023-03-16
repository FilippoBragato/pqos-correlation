import pickle
import numpy as np
import os
import tensorflow as tf
from . import datasetApi
import h5py

def load_and_preprocess_sample(address, serial_number):

    with h5py.File(address.numpy()) as f:
        main_group_key = list(f.keys())[0]
        time_group_key = str(serial_number.numpy()).zfill(5)
        array_data = f[main_group_key][time_group_key][()]

    # remove road
    array_data = array_data[array_data[:,2]>-0.9]

    # make image 200m*200m 0.2m
    img = np.zeros((1024,1024,2))
    img[:,:,1] = np.zeros((1024,1024)) - 0.9
    gt = np.zeros((1024,1024,1))
    for point in array_data:
        x_img = int(point[0]/0.16 + 512)
        y_img = int(point[1]/0.16 + 512)
        if x_img>=0 and x_img<1024 and y_img>=0 and y_img<1024:
            img[x_img, y_img, 0] += 1
            img[x_img, y_img, 1] += point[2]
            if point[3] != 0:
                gt[x_img, y_img, 0] = 1
    mask = img[:,:,0] > 0
    img[mask,1] = img[mask,1] / img[mask,0]
    img[:,:,1] = (img[:,:,1] + 1)/(10 + 1) 
    mask = img[:,:,0] < 10
    img[mask, 0] /= 10
    img[~mask, 0] = 1
    return img, gt

def create_tf_dataset(dataset:datasetApi.Dataset, specs, batch_size):
    addresses_train = []
    addresses_val = []
    addresses_test = []
    serials_train = []
    serials_val = []
    serials_test = []
    len_dataset_train = 0
    len_dataset_val = 0
    len_dataset_test = 0

    for spec in specs:

        length = dataset.get_measurement_series_length_TLC(spec[0], spec[1], spec[2], spec[3])
        length_train = int(np.floor(0.8* length))
        length_val = int(np.floor(0.1* length))
        length_test = int(np.floor(0.1* length))

        address_train = np.repeat([dataset._get_path_h5(spec[0], spec[1], spec[2], spec[3])], length_train, axis=0)
        addresses_train.append(address_train)
        address_val = np.repeat([dataset._get_path_h5(spec[0], spec[1], spec[2], spec[3])], length_val, axis=0)
        addresses_val.append(address_val)
        address_test = np.repeat([dataset._get_path_h5(spec[0], spec[1], spec[2], spec[3])], length_test, axis=0)
        addresses_test.append(address_test)

        serials_train.append(np.arange(length_train) + 1)
        serials_val.append(np.arange(length_val) + length_train + 1)
        serials_test.append(np.arange(length_test)+ length_train + length_val + 1)

        len_dataset_train += length_train
        len_dataset_val += length_val
        len_dataset_test += length_test

    ff = lambda ad, ser: tf.py_function(load_and_preprocess_sample, [ad, ser], [tf.float32, tf.float32])

    addresses_train = np.concatenate(addresses_train)
    serials_train = np.concatenate(serials_train)
    tf_dataset_train = tf.data.Dataset.from_tensor_slices((addresses_train, serials_train))
    tf_dataset_train = tf_dataset_train.shuffle(len_dataset_train)
    tf_dataset_train = tf_dataset_train.map(ff)
    tf_dataset_train = tf_dataset_train.cache("caches/train.cache")
    tf_dataset_train = tf_dataset_train.repeat()
    tf_dataset_train = tf_dataset_train.batch(batch_size=batch_size)
    tf_dataset_train = tf_dataset_train.prefetch(64)

    addresses_val = np.concatenate(addresses_val)
    serials_val = np.concatenate(serials_val)
    tf_dataset_val = tf.data.Dataset.from_tensor_slices((addresses_val, serials_val))
    tf_dataset_val = tf_dataset_val.map(ff)
    tf_dataset_val = tf_dataset_val.cache("caches/val.cache")
    tf_dataset_val = tf_dataset_val.repeat()
    tf_dataset_val = tf_dataset_val.batch(batch_size=batch_size)
    tf_dataset_val = tf_dataset_val.prefetch(64)

    addresses_test = np.concatenate(addresses_test)
    serials_test = np.concatenate(serials_test)
    tf_dataset_test = tf.data.Dataset.from_tensor_slices((addresses_test, serials_test))
    tf_dataset_test = tf_dataset_test.map(ff)
    tf_dataset_test = tf_dataset_test.cache("caches/test.cache")
    tf_dataset_test = tf_dataset_test.repeat()
    tf_dataset_test = tf_dataset_test.batch(batch_size=batch_size)
    tf_dataset_test = tf_dataset_test.prefetch(64)

    return tf_dataset_train, tf_dataset_val, tf_dataset_test, len_dataset_train, len_dataset_val, len_dataset_test

def conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 2, padding='same')(x)
    x = tf.keras.layers.Conv2D(n_filters, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(x)
    return x

def upconv_block(x, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 2, padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def get_model(input_shape):
    x_input = tf.keras.layers.Input(input_shape)
    x_d1 = conv_block(x_input, 16)
    x_d2 = conv_block(x_d1, 32)
    x_d3 = conv_block(x_d2, 64)
    x_d3 = tf.keras.layers.Dropout(0.5)(x_d3)
    x_u3 = upconv_block(x_d3, 32)
    x_u3 = tf.keras.layers.concatenate([x_d2, x_u3])
    x_u2 = upconv_block(x_u3, 16)
    x_u2 = tf.keras.layers.concatenate([x_d1, x_u2])
    x_u1 = upconv_block(x_u2, 8)

    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x_u1)

    model = tf.keras.Model(inputs=[x_input], outputs=[output], name="LightUNet")
    return model

def mini_conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 4, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4,4), strides=4, padding="same")(x)
    return x

def mini_upconv_block(x, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 4, padding='same', strides=4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def get_miniUNet(input_shape):
    x_input = tf.keras.layers.Input(input_shape)
    x_d1 = mini_conv_block(x_input, 4)
    x_d2 = mini_conv_block(x_d1, 8)
    x_u2 = mini_upconv_block(x_d2, 4)
    x_u2 = tf.keras.layers.concatenate([x_d1, x_u2])
    x_u1 = mini_upconv_block(x_u2, 8)

    output = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x_u1)

    model = tf.keras.Model(inputs=[x_input], outputs=[output], name="MiniUNet")
    return model
import pickle
import numpy as np
import os
import tensorflow as tf
from . import datasetApi
import h5py

def load_and_preprocess_sample(address, serial_number):

    with h5py.File(address[0].numpy()) as f:
        main_group_key = list(f.keys())[0]
        time_group_key = str(serial_number[0].numpy()).zfill(5)
        array_data = f[main_group_key][time_group_key][()]

    # remove road
    array_data = array_data[array_data[:,2]>-0.9]

    # make image 200m*200m 0.16m
    img = np.zeros((1250,1250,2))
    img[:,:,1] = np.zeros((1250,1250)) - 20
    gt = np.zeros((1250,1250,1))
    for point in array_data:
        x_img = int(point[0]/0.16 + 625)
        y_img = int(point[1]/0.16 + 625)
        img[x_img, y_img, 0] += 1
        img[x_img, y_img, 1] += point[2]
        if point[3] != 0:
            gt[x_img, y_img, 0] = 1
    mask = img[:,:,0] > 0
    img[mask,1] = img[mask,1] / img[mask,0]
    mask = img[:,:,0] < 10
    img[mask, 0] /= 10
    img[~mask, 0] = 1
    return img, gt

def create_tf_dataset(dataset:datasetApi.Dataset, specs):
    addresses = []
    serials = []
    len_dataset = 0
    for spec in specs:
        length = dataset.get_measurement_series_length_TLC(spec[0], spec[1], spec[2], spec[3])

        address = np.repeat([[dataset._get_path_h5(spec[0], spec[1], spec[2], spec[3])],], length, axis=0)
        addresses.append(address)
        serials.append(np.arange(length)[:, np.newaxis] + 1)
        len_dataset += length
    addresses = np.concatenate(addresses)
    serials = np.concatenate(serials)
    tf_dataset = tf.data.Dataset.from_tensor_slices((addresses, serials))
    tf_dataset = tf_dataset.shuffle(len_dataset)
    ff = lambda ad, ser: tf.py_function(load_and_preprocess_sample, [ad, ser], [tf.float32, tf.float32])
    tf_dataset = tf_dataset.map(ff)
    return tf_dataset


    


def _open_file_select_window(f, label, window):
    with open(f.numpy(), "rb") as fp:
        signal = pickle.load(fp)
    window_length = window[0]  # Number of samples per window
    stride_length = window[1]  # Number of samples to stride
    n_windows = window[2]
    starting_window = window[3]
    # windows creation
    window = np.arange(window_length)
    baseline = (np.arange(n_windows) + starting_window) * stride_length
    windows = np.tile(baseline, (window_length, 1)).T + window
    return np.abs(np.transpose(signal[:, windows, :], [1, 0, 2, 3])), np.repeat(
        [label], n_windows
    )


def _create_dataset_single(
    input_dir,
    signal_len,
    window_length,
    stride_length,
    batch_size,
    shuffle,
    cache_file,
    prefetch=True,
    repeat=True,
):
    n_windows = int(np.floor((signal_len - window_length) / stride_length)) + 1
    # int(np.floor(psutil.virtual_memory()[1])/(25000*window_length)) DUE to max usage of ram, can be actually increased
    max_windows = 300
    number_of_rep = int(np.ceil(n_windows / max_windows))
    starting = 0
    win_parameters = []
    for i in range(number_of_rep):
        if i == number_of_rep - 1:
            win_parameter = [
                window_length,
                stride_length,
                n_windows - starting,
                starting,
            ]
        else:
            win_parameter = [window_length,
                             stride_length, max_windows, starting]
        win_parameters.append(win_parameter)
        starting += max_windows
    files = os.listdir(input_dir)
    files.sort()
    labels = []
    temp = []
    for f in files:
        labels.append(int(f[:-4]))
        temp.append(input_dir + f)
    files = temp
    win_parameters_for_df = np.repeat(win_parameters, (len(labels)), axis=0)
    files = np.tile(files, number_of_rep)
    labels_fordf = np.tile(labels, number_of_rep)
    dataset_csi = tf.data.Dataset.from_tensor_slices(
        (files, labels_fordf, win_parameters_for_df)
    )
    if shuffle:
        dataset_csi = dataset_csi.shuffle(number_of_rep * len(labels))

    def map_func(files, labels_fordf, win_parameters_for_df): return (
        tf.py_function(
            func=_open_file_select_window,
            inp=[files, labels_fordf, win_parameters_for_df],
            Tout=[tf.float32, tf.int16],
        )
    )
    dataset_csi = dataset_csi.map(map_func=map_func)
    dataset_csi = dataset_csi.unbatch()
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(n_windows)
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(64)
    label_out = np.empty([0])
    first = True
    for params in win_parameters:
        n_wdw = params[2]
        if first:
            first = False
            label_out = np.repeat(labels, n_wdw)
        else:
            label_out = np.concatenate(
                [label_out, np.repeat(labels, n_wdw)], axis=0)
    return dataset_csi, label_out, n_windows * len(labels)


def create_dataset(input_dir, data_length, batch_size, window_length, stride_length):
    """ Creates tensorflow's datasets from the splitted dataset in input_dir

    Args:
        input_dir (str): path to the dataset as output of split_dataset
        data_length (int): the length of the signal of interest
        batch_size (int): length of the batch 
        window_length (int): length along the time axis for each sample
        stride_length (int): difference of positions along the time axis between two consecutive samples

    Returns:
        tf.data.Dataset: train dataset
        tf.data.Dataset: validation dataset
        tf.data.Dataset: test dataset
        dict: dictionary containing metadata about train, validation and test datasets
    """
    train_len = int(np.floor(data_length * 0.6))
    val_len = int(np.floor(data_length * 0.2))
    test_len = data_length - train_len - val_len

    name_base = "./dataset_cached/csi_net"
    name_cache_train = name_base + "_cache_train"

    dataset_csi_train, label_train, num_samples_train = _create_dataset_single(
        input_dir + "train/",
        train_len,
        window_length,
        stride_length,
        batch_size,
        shuffle=True,
        cache_file=name_cache_train
    )

    name_cache_val = name_base + "_cache_val"
    dataset_csi_val, label_val, num_samples_val = _create_dataset_single(
        input_dir + "validation/",
        val_len,
        window_length,
        stride_length,
        batch_size,
        shuffle=False,
        cache_file=name_cache_val
    )

    name_cache_test = name_base + "_cache_test"
    dataset_csi_test, label_test, num_samples_test = _create_dataset_single(
        input_dir + "test/",
        test_len,
        window_length,
        stride_length,
        batch_size,
        shuffle=False,
        cache_file=name_cache_test
    )

    info = {
        "num_samples_train": num_samples_train,
        "num_samples_val": num_samples_val,
        "num_samples_test": num_samples_test,
        "labels_train_selected_expanded": label_train,
        "labels_val_selected_expanded": label_val,
        "labels_test_selected_expanded": label_test,
        "labels_considered": np.unique(label_train),
    }
    return dataset_csi_train, dataset_csi_val, dataset_csi_test, info

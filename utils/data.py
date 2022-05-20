import glob
import os

import tensorflow as tf
import tensorflow_datasets as tfds

split_map = {
    "svhn_cropped": ["train + extra", "test[0%:50%]", "test[50%:100%]"],
    "cifar10": ["train", "test[0%:50%]", "test[50%:100%]"],
    "mnist": ["train", "test", "test"],
    "celeba": ["train", "validation"],
}


def normalize(img, label):
    return tf.cast((img), tf.float32) / 255.0, label


def setup_data(
    dataset="svhn_cropped", data_dir=None, batch_size=128, val_batch_size=500
):

    if dataset == "celeba":
        ds_train, ds_val, ds_test = get_celeba()
    else:
        ds_train, ds_val, ds_test = get_tfds(dataset, data_dir)

    # https://stackoverflow.com/a/50453698
    # https://stackoverflow.com/a/49916221
    ds_train = (
        ds_train.map(normalize, num_parallel_calls=4)
        .shuffle(len(ds_train))
        .repeat()
        .batch(batch_size)
        .prefetch(4)
    )
    ds_val = (
        ds_val.map(normalize, num_parallel_calls=4)
        .repeat()
        .batch(val_batch_size)
        .prefetch(4)
    )

    ds_test = ds_test.map(normalize).prefetch(4)

    return iter(ds_train), iter(ds_val), ds_test


def get_tfds(dataset, data_dir):
    data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        dataset,
        split=split_map[dataset],
        shuffle_files=True,
        data_dir=data_dir,
        with_info=True,
        as_supervised=True,
    )

    return ds_train, ds_val, ds_test


def parse_tfrecord_tf(record):
    """
    wget https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
    tar -xvf celeb-tfr.tar

    https://github.com/openai/glow
    https://github.com/openai/glow/blob/master/data_loaders/get_data.py#L10
    """
    features = tf.io.parse_single_example(
        record,
        features={
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "data": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        },
    )

    data, label, shape = features["data"], features["label"], features["shape"]
    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
    img = tf.io.decode_raw(data, tf.uint8)

    img = tf.reshape(img, shape)
    # img = tf.image.central_crop(img, 0.6)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [64, 64])
    return img, label


def get_celeba():

    DS = []
    for split in ["train", "validation"]:

        filenames = glob.glob(f"data/celeba-tfr/{split}/*")
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # https://github.com/openai/glow/blob/master/data_loaders/get_data.py#L42
        dataset = raw_dataset.map(lambda x: parse_tfrecord_tf(x), num_parallel_calls=4)
        DS.append(dataset)

    return DS[0], DS[1], DS[1]  # No test-set here

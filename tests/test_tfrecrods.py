import glob
import os

import tensorflow as tf

if __name__ == "__main__":
    """
    wget https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
    tar -xvf celeb-tfr.tar
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    filenames = glob.glob("data/celeba-tfr/train/*")
    raw_dataset = tf.data.TFRecordDataset(filenames)
    ds = iter(raw_dataset)
    res = next(ds)
    # So that's not how to do it!

    # https://github.com/openai/glow/blob/master/data_loaders/get_data.py#L10
    def parse_tfrecord_tf(record):
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
        img = tf.image.central_crop(img, 0.6)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.resize(img, [64, 64])
        return img, label

    # https://github.com/openai/glow/blob/master/data_loaders/get_data.py#L42
    dataset = raw_dataset.map(lambda x: parse_tfrecord_tf(x), num_parallel_calls=4)
    ds = iter(dataset)
    res = next(ds)

    import matplotlib.pyplot as plt

    plt.imshow(res[0] / 255)
    plt.show()

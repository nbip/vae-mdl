import os

import tensorflow as tf
import tensorflow_datasets as tfds

import utils


def setup_data(data_dir=None):
    def normalize(img, label):
        prob = tf.cast(img, tf.float32) / 255.0
        img = utils.bernoullisample(prob)
        return img, label

    batch_size = 500
    data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test[0%:50%]", "test[50%:100%]"],
        shuffle_files=True,
        data_dir=data_dir,
        with_info=True,
        as_supervised=True,
    )

    # ---- shuffling, infinite yield, preprocessing
    # https://stackoverflow.com/a/50453698
    # https://stackoverflow.com/a/49916221
    # https://www.tensorflow.org/guide/data#randomly_shuffling_input_data
    # https://www.tensorflow.org/datasets/overview
    # https://www.tensorflow.org/datasets/splits
    # manual dataset https://www.tensorflow.org/datasets/add_dataset
    # https://www.reddit.com/r/MachineLearning/comments/65me2d/d_proper_crop_for_celeba/
    # celeba hint: https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/train.py#L121
    # download celeba: https://github.com/AntixK/PyTorch-VAE
    ds_train = (
        ds_train.map(normalize, num_parallel_calls=4)
        .shuffle(len(ds_train))
        .repeat(2)
        .batch(batch_size)
        .prefetch(4)
    )
    ds_val = (
        ds_val.map(normalize, num_parallel_calls=4)
        .repeat(2)
        .batch(batch_size)
        .prefetch(4)
    )

    ds_test = ds_test.map(normalize).prefetch(4)

    return iter(ds_train), iter(ds_val), ds_test


def setup_data2(data_dir=None):
    batch_size = 500
    data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    # ---- load data
    (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]

    Xtrain = Xtrain[..., None] / 255.0
    Xtest = Xtest[..., None] / 255.0

    Xtrain_binarized = utils.bernoullisample(Xtrain)
    Xtest_binarized = utils.bernoullisample(Xtest)

    ds_train = (
        tf.data.Dataset.from_tensor_slices((Xtrain_binarized, ytrain))
        .shuffle(Ntrain)
        .repeat(2)
        .batch(batch_size)
        .prefetch(4)
    )
    ds_test = (
        tf.data.Dataset.from_tensor_slices((Xtest_binarized, ytest))
        .repeat(2)
        .batch(batch_size)
        .prefetch(4)
    )

    return iter(ds_train), iter(ds_test), ds_test


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train_loader, val_loader, ds_test = setup_data()

    for x, y in val_loader:
        fig, ax = plt.subplots()
        ax.imshow(x[0], cmap="Greys")
        plt.show()
        plt.close()

    train_loader, val_loader, ds_test = setup_data2()

    i = 0
    for x, y in val_loader:
        i += 1
        print(i)
        fig, ax = plt.subplots()
        ax.imshow(x[0], cmap="Greys")
        plt.show()
        plt.close()

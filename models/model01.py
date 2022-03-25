"""
https://github.com/rasmusbergpalm/vnca/blob/dmg-double-celeba/vae-nca.py
https://github.com/rasmusbergpalm/vnca/tree/dmg_celebA_baseline
https://github.com/rasmusbergpalm/vnca/blob/dmg_celebA_baseline/modules/vae.py#L88

TODO: bits_pr_dim or other scaling of loss
https://github.com/rasmusbergpalm/vnca/blob/main/modules/loss.py
https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
"""

import os
from datetime import datetime

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from models.loss import iwae_loss
from models.model import Model
from utils import MixtureDiscretizedLogistic, logmeanexp


def Conv2D(*args, **kwargs):
    return conv2DWrap(*args, transpose=False, **kwargs)


def Conv2DTranspose(*args, **kwargs):
    return conv2DWrap(*args, transpose=True, **kwargs)


class conv2DWrap(tf.keras.layers.Layer):
    """
    Wrapper around convolutional operations to allow for multiple leading dimensions.

    Example:
    in [samples, batch, h, w, c] -> intermediate [samples * batch, ...] -> out [samples, batch, h2, w2, c2]
    """

    def __init__(self, *args, transpose=False, **kwargs):
        super(conv2DWrap, self).__init__()

        self.conv = (
            tf.keras.layers.Conv2DTranspose(*args, **kwargs)
            if transpose
            else tf.keras.layers.Conv2D(*args, **kwargs)
        )

    def call(self, x, **kwargs):
        in_shape = x.shape  # [samples, batch, h, w, c] or [batch, h, w, c]

        # --- merge sample and batch dim
        x = tf.reshape(x, [-1, *in_shape[-3:]])

        # ---- do the conv
        out = self.conv(x)

        # ---- unmerge sample and batch dim
        out_shape = out.shape
        out = tf.reshape(out, [*in_shape[:-3], *out_shape[-3:]])

        return out


class Model01(Model, tf.keras.Model):
    def __init__(self):
        super(Model01, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 10
        self.global_step = 0

        self.loss_fn = iwae_loss

        self.train_loader, self.val_loader = self.setup_data()

        self.encoder = tf.keras.Sequential(
            [
                Conv2D(
                    32, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu
                ),
                Conv2D(
                    64, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu
                ),
                Conv2D(128, kernel_size=5, strides=2, padding="same", activation=None),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    64, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu
                ),
                Conv2DTranspose(
                    32, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu
                ),
                Conv2DTranspose(
                    100, kernel_size=5, strides=2, padding="same", activation=None
                ),
            ]
        )

    def call(self, x, n_samples=1, **kwargs):
        qzx = self.encode(x)
        z = qzx.sample(n_samples)
        pxz = self.decode(z)
        return z, qzx, pxz

    def encode(self, x):
        q = self.encoder(x)
        loc, logscale = tf.split(q, num_or_size_splits=2, axis=-1)
        return tfd.Normal(loc, tf.exp(logscale) + 1e-6)

    def decode(self, z):
        logits = self.decoder(z)
        return MixtureDiscretizedLogistic(logits)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, qzx, pxz = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(z, qzx, x, pxz)

        grads = tape.gradient(loss, self.trainable_weights)
        # grads = [tf.clip_by_norm(g, clip_norm=1.0) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, x):
        z, qzx, pxz = self(x)
        loss, metrics = self.loss_fn(z, qzx, x, pxz)
        return loss, metrics

    def train_batch(self):
        x, y = next(self.train_loader)
        loss, metrics = self.train_step(x)

        return loss, metrics

    def val_batch(self):
        x, y = next(self.val_loader)
        loss, metrics = self.val_step(x)
        return loss, metrics

    def report(self, writer, metrics):
        pass

    def setup_data(self, data_dir=None):
        def normalize(img, label):
            return tf.cast((img), tf.float32) / 255.0, label

        batch_size = 128
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        import tensorflow_datasets as tfds

        (ds_train, ds_test), ds_info = tfds.load(
            "cifar10",
            split=["train", "test"],
            shuffle_files=True,
            data_dir=data_dir,
            with_info=True,
            as_supervised=True,
        )

        # TODO: shuffle, infinite yield, preprocessing
        # https://stackoverflow.com/a/50453698
        # https://stackoverflow.com/a/49916221
        ds_train = (
            ds_train.map(
                normalize, num_parallel_calls=4).shuffle(50000).repeat().batch(batch_size).prefetch(4)
        )
        ds_test = (
            ds_test.map(normalize, num_parallel_calls=4).batch(batch_size).prefetch(4)
        )

        return iter(ds_train), iter(ds_test)

    def save(self, fp):
        pass

    def load(self, fp):
        pass

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        revision = os.environ.get("REVISION") or datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        train_log_dir = f"/tmp/{experiment}/{revision}/train"
        test_log_dir = f"/tmp/{experiment}/{revision}/test"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{experiment}/{revision}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":

    import os
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    b, h, w, c = 5, 32, 32, 3
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    model = Model01()

    # x, y = next(model.train_loader)
    # for i in range(500):
    #     print(i)
    #     x, y = next(model.train_loader)
    #     print(x.shape)
    #     if x.shape[0] != 128:
    #         break

    import matplotlib; matplotlib.use('Agg')  # needed when running from commandline
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(x[0])
    plt.show()
    plt.savefig('img')
    plt.close()

    # ---- test model subparts
    qzx = model.encode(x)
    z = qzx.sample(model.n_samples)
    pxz = model.decode(z)
    z, qzx, pxz = model(x, model.n_samples)

    # ---- test model reconstructions
    x = x[0][None, :]
    qzx = model.encode(x)
    z = qzx.sample(model.n_samples)
    pxz = model.decode(z)
    z, qzx, pxz = model(x, model.n_samples)
    x_samples = pxz.sample(100)
    mean = tf.reduce_mean(x_samples, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(mean[0, 0, :])
    plt.show()
    plt.close()

    model.train_step(x)

    model.train_batch()
    model.val_batch()

    for i in range(100_000):
        loss, metrics = model.train_batch()
        if i % 100 == 0:
            print(i, loss)

    # ---- test model reconstructions
    x = x[0][None, :]
    qzx = model.encode(x)
    z = qzx.sample(model.n_samples)
    pxz = model.decode(z)
    z, qzx, pxz = model(x, model.n_samples)
    x_samples = pxz.sample(100)
    mean = tf.reduce_mean(x_samples, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(mean[0, 0, :])
    plt.show()
    plt.savefig('img_rec')
    plt.close()

    # conv = conv2DWrap(32, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu)

    # x = np.random.rand(10, b, h, w, c).astype(np.float32)
    # out = conv(x)
    # print(out.shape)

    # conv_t = conv2DWrap(3, transpose=True, kernel_size=5, strides=2, padding="same", activation=tf.nn.elu)

    # reversed = conv_t(out)
    # print(reversed.shape)

    # # ---- shuffle before repeat?
    # # shuffle then repeat makes sure every element is seen before a new epoch
    # ds = tf.data.Dataset.range(6)
    #
    # # ds = ds.repeat()
    # # ds = ds.shuffle(6)
    # ds = ds.shuffle(100000)
    # ds = ds.repeat()
    # ds = ds.batch(2)
    #
    # iterator = iter(ds)
    # for i in range(20):
    #     if i % (10 // 2) == 0:
    #         print("------------")
    #     print("{:02d}:".format(i), next(iterator))

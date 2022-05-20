"""
Same as model09, but for mnist
look at svhn/task32.py
"""
import os
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from models.loss import iwae_loss
from models.model import Model
from utils import (
    MixtureDiscretizedLogistic,
    PixelMixtureDiscretizedLogistic,
    logmeanexp,
)


class GatedBlock(layers.Layer):
    """GLU: https://arxiv.org/pdf/1612.08083.pdf"""

    def __init__(self, filters=64, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.Sequential(
            [
                layers.Conv2D(
                    filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                layers.Conv2D(
                    2 * filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation=None,
                ),
            ]
        )

    def call(self, inputs, **kwargs):
        A, B = tf.split(self.conv(inputs), 2, axis=-1)
        H = A * tf.nn.sigmoid(B)
        return tf.nn.relu(H)


class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.Sequential(
            [
                layers.Conv2D(
                    128, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(
                    256, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(
                    256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
                ),
                *[GatedBlock() for _ in range(5)],
            ]
        )

        self.dense = layers.Dense(2 * 20, activation=None)

    def call(self, x, **kwargs):

        h = self.conv(x)  # [batch, 7, 7, 64]
        h = tf.reshape(h, [-1, 7 * 7 * 64])
        return self.dense(h)


class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dense = layers.Dense(7 * 7 * 64, activation=tf.nn.tanh)

        self.convt = tf.keras.Sequential(
            [
                layers.Conv2D(
                    256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
                ),
                *[GatedBlock() for _ in range(5)],
                layers.Conv2DTranspose(
                    128, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2DTranspose(
                    2, kernel_size=4, strides=2, padding="same", activation=None
                ),
            ]
        )

    def call(self, z, **kwargs):

        shape = z.shape

        # ---- batch and samples in same dimension
        z = tf.reshape(z, [-1, shape[-1]])

        h = self.dense(z)
        h = tf.reshape(h, [-1, 7, 7, 64])

        h = self.convt(h)

        # ---- unmerge batch and samples
        out_shape = h.shape
        h = tf.reshape(h, [*shape[:-1], *out_shape[-3:]])

        return h


class Model10(Model, tf.keras.Model):
    def __init__(self):
        super(Model10, self).__init__()

        self.optimizer = tf.keras.optimizers.Adamax(1e-4)
        self.n_samples = 5
        self.global_step = 0
        self.init_tensorboard()

        self.loss_fn = iwae_loss

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1]

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.train_loader, self.val_loader, self.ds_test = self.setup_data()

    def call(self, x, n_samples=1, **kwargs):
        qzx = self.encode(x)
        z = qzx.sample(n_samples)
        pxz = self.decode(z)
        return z, qzx, pxz

    def encode(self, x):
        q = self.encoder(x)
        loc, logscale = tf.split(q, num_or_size_splits=2, axis=-1)
        qzx = tfd.Normal(loc, tf.nn.softplus(logscale) + 1e-6)
        qzx.axes = [-1]  # specify axes to sum over in log_prob
        return qzx

    def decode(self, z):
        logits = self.decoder(z)
        pxz = tfd.Bernoulli(logits=logits)
        pxz.axes = [-1, -2, -3]  # specify axes to sum over in log_prob
        return pxz

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, qzx, pxz = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(x, z, self.pz, qzx, pxz)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, x):
        z, qzx, pxz = self(x, n_samples=self.n_samples)
        loss, metrics = self.loss_fn(x, z, self.pz, qzx, pxz)
        return loss, metrics

    def train_batch(self):
        x, y = next(self.train_loader)
        loss, metrics = self.train_step(x)
        self.global_step += 1

        return loss, metrics

    def val_batch(self):
        x, y = next(self.val_loader)
        loss, metrics = self.val_step(x)
        self.report(x, metrics)
        return loss, metrics

    def test(self):
        # TODO: test: https://github.com/rasmusbergpalm/vnca/blob/dmg_celebA_baseline/modules/vae.py#L124
        pass

    def report(self, x, metrics):
        samples, recs = self._plot_samples(x)

        with self.val_summary_writer.as_default():
            tf.summary.image("Evaluation/img", x[0][None, :], step=self.global_step)
            tf.summary.image("Evaluation/img_rec", recs[0, :], step=self.global_step)
            tf.summary.image(
                "Evaluation/img_samp", samples[0, :], step=self.global_step
            )
            for key, value in metrics.items():
                tf.summary.scalar(
                    f"Evalutation/{key}", value.numpy().mean(), step=self.global_step
                )

    def _plot_samples(self, x):
        z, qzx, pxz = self(x[0][None, :], n_samples=self.n_samples)
        recs = pxz.mean()  # [n_samples, batch, h, w, ch]
        # recs = pxz.mean(n=100)  # [n_samples, batch, h, w, ch]

        pz = tfd.Normal(tf.zeros_like(z), tf.ones_like(z))
        pxz = self.decode(pz.sample())
        samples = tf.cast(pxz.sample(), tf.float32)  # [n_samples, batch, h, w, ch]

        return samples, recs

    def save(self, fp):
        self.save_weights(f"{fp}_10")

    def load(self, fp):
        self.load_weights(f"{fp}_10")

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        revision = os.environ.get("REVISION") or datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        train_log_dir = f"/tmp/{experiment}/{revision}/train"
        val_log_dir = f"/tmp/{experiment}/{revision}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{experiment}/{revision}"
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def setup_data(data_dir=None):
        def normalize(img, label):
            prob = tf.cast(img, tf.float32) / 255.0
            img = tf.cast(
                prob > tf.random.stateless_uniform(prob.shape, seed=(4, 2)), tf.float32
            )
            # TODO: for some reason you need to use stateless unform here, but then the random numbers are the same every time, so a kind of static binarization
            return img, label

        batch_size = 128
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
            .repeat()
            .batch(batch_size)
            .prefetch(4)
        )
        ds_val = (
            ds_val.map(normalize, num_parallel_calls=4)
            .repeat()
            .batch(batch_size)
            .prefetch(4)
        )

        ds_test = ds_test.map(normalize).prefetch(4)

        return iter(ds_train), iter(ds_val), ds_test


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib.pyplot as plt
    import numpy as np

    b, h, w, c = 5, 32, 32, 3
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    model = Model10()
    # model.load("best")

    x, y = next(model.train_loader)

    z, qzx, pxz = model(x, model.n_samples)

    for i in range(100_000):
        train_loss, train_metrics = model.train_batch()
        if i % 100 == 0:
            val_loss, val_metrics = model.val_batch()
            print(f"{i}, train loss {train_loss:.2f}, val loss {val_loss:.2f}")

    x, y = next(model.train_loader)
    x = x[0][None, :]
    z, qzx, pxz = model(x, model.n_samples)
    mean = pxz.mean()

    fig, ax = plt.subplots()
    ax.imshow(x[0])
    plt.show()
    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(mean[0, 0, :])
    plt.show()
    plt.close()
    samples, rec = model._plot_samples(x)
    fig, ax = plt.subplots()
    ax.imshow(samples[0, 0, :])
    plt.show()
    plt.close()

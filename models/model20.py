"""
As model19.py, but with Normal loss with tanh on logstd, instead of mse.
Great samples and reconstructions, learning is quick, but then no more gains.
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from models.loss import iwae_loss
from models.model import Model


class GlobalStep(object):
    """
    https://stackoverflow.com/a/6192298
    https://codereview.stackexchange.com/q/253675
    """

    def __init__(self):
        self._value = 0
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        for callback in self._observers:
            # print('announcing change')
            callback(self._value)

    def bind_to(self, callback):
        # print('bound')
        self._observers.append(callback)


class DataSets:
    def __init__(self):
        self.train_loader, self.val_loader, self.ds_test = self.setup_data()

    @staticmethod
    def setup_data(data_dir=None):
        def normalize(img, label):
            return tf.cast((img), tf.float32) / 255.0, label

        batch_size = 128
        val_batch_size = 500
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            # "cifar10",
            "svhn_cropped",
            # split=["train", "test[0%:50%]", "test[50%:100%]"],
            split=["extra", "test[0%:50%]", "test[50%:100%]"],
            shuffle_files=True,
            data_dir=data_dir,
            with_info=True,
            as_supervised=True,
        )

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


class Encoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()
        self.n_latent = n_latent

        self.convs = tf.keras.Sequential(
            [
                layers.Conv2D(
                    32, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(
                    64, strides=2, kernel_size=3, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(
                    128, strides=2, kernel_size=3, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(
                    256, strides=2, kernel_size=3, padding="same", activation=tf.nn.relu
                ),
            ]
        )

        conv_out_dim = 32 // 2 ** 3 * 32 // 2 ** 3 * 256
        self.fc = layers.Dense(2 * n_latent)

    def call(self, x, **kwargs):
        out = self.convs(x)
        out = tf.reshape(out, [out.shape[0], -1])
        mu, logstd = tf.split(self.fc(out), num_or_size_splits=2, axis=-1)
        return tfd.Normal(mu, tf.nn.softplus(logstd))


class Decoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()
        self.n_latent = n_latent
        self.n_mix = 5
        output_shape = (32, 32, 3)

        self.base_size = [output_shape[0] // 2 ** 3, output_shape[1] // 2 ** 3, 128]
        self.fc = layers.Dense(np.prod(self.base_size), activation=tf.nn.relu)

        self.deconvs = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(
                    128, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2DTranspose(
                    64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2DTranspose(
                    32, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu
                ),
                layers.Conv2D(3 * 2, kernel_size=3, padding="same", activation=None),
            ]
        )

    def call(self, z, **kwargs):
        h = self.fc(z)
        # ---- merge sample and batch dimensions and reshape from dense to conv, mirrored from encoder
        h = tf.reshape(h, [-1, *self.base_size])
        out = self.deconvs(h)
        out = tf.reshape(out, [*z.shape[:-1], 32, 32, 3 * 2])
        mu, logstd = tf.split(out, num_or_size_splits=2, axis=-1)
        pxz = tfd.Normal(mu, tf.exp(tf.nn.tanh(logstd)))
        return pxz


class Model20(Model, tf.keras.Model):
    def __init__(self):
        super(Model20, self).__init__()

        # self.optimizer = tf.keras.optimizers.Adamax(1e-3)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 5
        self.global_step = GlobalStep()
        self.global_step.bind_to(
            self.update_learning_rate
        )  # add callback to update learning rate when gs.value is changed
        self.init_tensorboard()

        self.loss_fn = iwae_loss
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1]

        self.encoder = Encoder(16)
        self.decoder = Decoder(16)

        self.ds = DataSets()

    def update_learning_rate(self, value):
        if value in [2 ** i * 7000 for i in range(8)]:
            old_lr = self.optimizer.learning_rate.numpy()
            new_lr = 1e-3 * 10 ** (-value / (2 ** 7 * 7000))
            self.optimizer.learning_rate.assign(new_lr)
            print(f"Changing learningrate from {old_lr:.2e} to {new_lr:.2e}")

    def call(self, x, n_samples=1, **kwargs):
        qzx = self.encode(x)
        z = qzx.sample(n_samples)
        mu = self.decode(z)
        return z, qzx, mu

    def encode(self, x):
        qzx = self.encoder(x)
        qzx.axes = [-1]  # specify axes to sum over in log_prob
        return qzx

    def decode(self, z):
        pxz = self.decoder(z)
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
        x, y = next(self.ds.train_loader)
        loss, metrics = self.train_step(x)
        self.global_step.value += 1
        return loss, metrics

    def val_batch(self):
        x, y = next(self.ds.val_loader)
        loss, metrics = self.val_step(x)
        self.report(x, metrics)
        return loss, metrics

    def test(self, n_samples):
        llh = np.nan * np.zeros(len(self.ds.ds_test))

        for i, (x, y) in enumerate(tqdm(self.ds.ds_test)):
            z, qzx, pxz = self(x[None, :], n_samples=n_samples)
            loss, metrics = self.loss_fn(x, z, self.pz, qzx, pxz)
            llh[i] = metrics["iwae_elbo"]

        return llh.mean(), llh

    def report(self, x, metrics):
        samples, recs = self._plot_samples(x)

        with self.val_summary_writer.as_default():
            tf.summary.image(
                "Evaluation/img", x[0][None, :], step=self.global_step.value
            )
            tf.summary.image(
                "Evaluation/img_rec", recs[None, :], step=self.global_step.value
            )
            tf.summary.image(
                "Evaluation/img_samp", samples[None, :], step=self.global_step.value
            )
            for key, value in metrics.items():
                tf.summary.scalar(
                    f"Evalutation/{key}",
                    value.numpy().mean(),
                    step=self.global_step.value,
                )

    def _plot_samples(self, x):
        n, h, w, c = 8, 32, 32, 3
        z, qzx, pxz = self(x[: n ** 2], n_samples=self.n_samples)
        recs = pxz.mean()[0]  # [n_samples, batch, h, w, ch]

        canvas1 = np.random.rand(n * h, n * w, c)
        for i in range(n):
            for j in range(n):
                canvas1[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = recs[
                    i * n + j, :, :, :
                ]

        pz = tfd.Normal(tf.zeros_like(z), tf.ones_like(z))
        pxz = self.decode(pz.sample())
        # samples = tf.cast(pxz.sample(), tf.float32)[0]  # [n_samples, batch, h, w, ch]
        samples = tf.cast(pxz.mean(), tf.float32)[0]  # [n_samples, batch, h, w, ch]

        canvas2 = np.random.rand(n * h, n * w, c)
        for i in range(n):
            for j in range(n):
                canvas2[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = samples[
                    i * n + j, :, :, :
                ]

        return canvas2, canvas1

    def save(self, fp):
        self.save_weights(f"{fp}_20")

    def load(self, fp):
        self.load_weights(f"{fp}_20")

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


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model20.py > models/model20.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model20()

    # intialize model
    model.val_batch()

    train(model, n_updates=1_400_000, eval_interval=1000)

    model.load("best")
    mean_llh, llh = model.test(5000)

    print(mean_llh)

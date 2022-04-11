"""
Reproduce IWAE results on statically binarized mnist
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

import utils
from models.loss import iwae_loss
from models.model import Model


class LearningRate(object):
    """https://stackoverflow.com/a/6192298"""
    def __init__(self):
        self._learning_rate = 0.001
        self._observers = []

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
        for callback in self._observers:
            print('announcing change')
            callback(self._learning_rate)

    def bind_to(self, callback):
        print('bound')
        self._observers.append(callback)


class DataSets:
    def __init__(self):

        self.train_loader, self.val_loader, self.ds_test = self.setup_data()

    @staticmethod
    def setup_data(data_dir=None):
        def normalize(img, label):
            prob = tf.cast(img, tf.float32) / 255.0
            img = utils.bernoullisample(prob, seed=42)
            return img, label

        batch_size = 128
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        # https://stackoverflow.com/a/50453698
        # https://stackoverflow.com/a/49916221
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist",
            split=["train", "test", "test"],
            shuffle_files=True,
            data_dir=data_dir,
            with_info=True,
            as_supervised=True,
        )

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
            .batch(len(ds_val))
            .prefetch(4)
        )

        ds_test = ds_test.map(normalize).prefetch(4)

        return iter(ds_train), iter(ds_val), ds_test


class BasicBlock(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = layers.Dense(n_latent, activation=None)
        self.lstd = layers.Dense(n_latent, activation=tf.exp)

    def call(self, input, **kwargs):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        qz_given_input = tfd.Normal(q_mu, q_std + 1e-6)

        return qz_given_input


class Encoder(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encode_x_to_z = BasicBlock(n_hidden, n_latent)

    def call(self, x, **kwargs):
        x = tf.reshape(x, [x.shape[0], -1])
        qzx = self.encode_x_to_z(x)
        return qzx


class Decoder(tf.keras.Model):
    def __init__(self, n_hidden, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.decode_z_to_x = tf.keras.Sequential(
            [
                layers.Dense(n_hidden, activation=tf.nn.tanh),
                layers.Dense(n_hidden, activation=tf.nn.tanh),
                layers.Dense(784, activation=None),
            ]
        )

    def call(self, z, **kwargs):
        logits = self.decode_z_to_x(z)
        logits = tf.reshape(logits, [*z.shape[:2], 28, 28, 1])
        pxz = tfd.Bernoulli(logits=logits)
        return pxz


class Model11(Model, tf.keras.Model):
    def __init__(self):
        super(Model11, self).__init__()

        self.optimizer = tf.keras.optimizers.Adamax(1e-4)
        self.n_samples = 5
        self.global_step = 0
        self.init_tensorboard()

        self.loss_fn = iwae_loss

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1]

        self.encoder = Encoder(200, 100)
        self.decoder = Decoder(200)

        self.ds = DataSets()

        self._observers = []
        self.learning_rate = LearningRate()
        self.learning_rate.bind_to(self.update_learning_rate)

    def bind_to(self, callback):
        print('bound')
        self._observers.append(callback)

    def update_learning_rate(self, global_step):
        print("Changing learningrate")

    def call(self, x, n_samples=1, **kwargs):
        qzx = self.encode(x)
        z = qzx.sample(n_samples)
        pxz = self.decode(z)
        return z, qzx, pxz

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
        self.global_step += 1
        return loss, metrics

    def val_batch(self):
        x, y = next(self.ds.val_loader)
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
            tf.summary.image("Evaluation/img_rec", recs[None, :], step=self.global_step)
            tf.summary.image(
                "Evaluation/img_samp", samples[None, :], step=self.global_step
            )
            for key, value in metrics.items():
                tf.summary.scalar(
                    f"Evalutation/{key}", value.numpy().mean(), step=self.global_step
                )

    def _plot_samples(self, x):
        n, h, w, c = 8, 28, 28, 1
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
        samples = tf.cast(pxz.sample(), tf.float32)[0]  # [n_samples, batch, h, w, ch]

        canvas2 = np.random.rand(n * h, n * w, c)
        for i in range(n):
            for j in range(n):
                canvas2[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = samples[
                    i * n + j, :, :, :
                ]

        return canvas2, canvas1

    def save(self, fp):
        self.save_weights(f"{fp}_11")

    def load(self, fp):
        self.load_weights(f"{fp}_11")

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
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model11.py > models/model11.log &
    from trainer import train

    model = Model11()
    model.val_batch()
    train(model, n_updates=1_000_000, eval_interval=1000)

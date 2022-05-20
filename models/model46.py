"""
As model43 but only conv blocks and no tanh on logstd. sqrt(1/nblocks) weight init.
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from models.loss import iwae_loss
from models.model import Model
from modules import DecoderBlock, EncoderBlock
from utils import DiscretizedLogistic, GlobalStep, fill_canvas, setup_data


class DataSets:
    def __init__(self):
        self.train_loader, self.val_loader, self.ds_test = setup_data(
            "svhn_cropped", batch_size=64, val_batch_size=64
        )


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        n_latent = 20
        filters = 64
        hidden = 32
        n_blocks = 5
        in_shape = (32, 32, 3)
        conv_out_shape = [in_shape[0] // 2 ** 3, in_shape[1] // 2 ** 3, filters]

        self.convs = tf.keras.Sequential(
            [
                layers.Conv2D(
                    filters,
                    strides=1,
                    kernel_size=3,
                    padding="same",
                    activation=tf.nn.gelu,
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
                ),
                layers.Conv2D(
                    2 * filters,
                    strides=1,
                    kernel_size=3,
                    padding="same",
                    activation=tf.nn.gelu,
                ),
            ]
        )

    def call(self, x, **kwargs):
        mu, logstd = tf.split(self.convs(x), num_or_size_splits=2, axis=-1)
        return tfd.Normal(mu, tf.nn.softplus(logstd))


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out_shape = (32, 32, 3)
        filters = 64
        hidden = 32
        n_blocks = 5

        self.deconvs = tf.keras.Sequential(
            [
                # layers.Conv2D(
                #     filters, strides=1, kernel_size=3, padding="same", activation=tf.nn.gelu
                # ),
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                layers.Conv2D(
                    3 * 2, strides=1, kernel_size=3, padding="same", activation=None
                ),
            ]
        )

    def call(self, z, **kwargs):
        # merge sample and batch dimension
        shape = z.shape
        z = tf.reshape(z, [-1, *shape[-3:]])
        out = self.deconvs(z)

        # unmerge sample and batch dimension
        out = tf.reshape(out, [*shape[:-3], *out.shape[-3:]])

        mu, logstd = tf.split(out, num_or_size_splits=2, axis=-1)
        pxz = DiscretizedLogistic(mu, logstd, low=0.0, high=1.0, levels=256)
        return pxz


class Model46(Model, tf.keras.Model):
    def __init__(self):
        super(Model46, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 5
        self.global_step = GlobalStep()
        self.global_step.bind_to(
            self.update_learning_rate
        )  # add callback to update learning rate when gs.value is changed
        self.init_tensorboard()

        self.loss_fn = iwae_loss

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1, -2, -3]

        self.encoder = Encoder()
        self.decoder = Decoder()

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
        pxz = self.decode(z)
        return z, qzx, pxz

    def encode(self, x):
        qzx = self.encoder(x)
        qzx.axes = [-1, -2, -3]  # specify axes to sum over in log_prob
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
        samples, recs, imgs = self._plot_samples(x)

        with self.val_summary_writer.as_default():
            tf.summary.image(
                "Evaluation/images", imgs[None, :], step=self.global_step.value
            )
            tf.summary.image(
                "Evaluation/reconstructions", recs[None, :], step=self.global_step.value
            )
            tf.summary.image(
                "Evaluation/generative-samples",
                samples[None, :],
                step=self.global_step.value,
            )
            for key, value in metrics.items():
                tf.summary.scalar(
                    f"Evalutation/{key}",
                    value.numpy().mean(),
                    step=self.global_step.value,
                )

    def _plot_samples(self, x):
        n, h, w, c = 8, 32, 32, 3

        # reconstructions
        z, qzx, pxz = self(x[: n ** 2], n_samples=1)
        recs = pxz.mean()[0]  # [n_samples, batch, h, w, ch]

        # samples
        pz = tfd.Normal(tf.zeros_like(z), tf.ones_like(z))
        pxz = self.decode(pz.sample())
        samples = np.clip(pxz.mean()[0], 0.0, 1.0)  # [n_samples, batch, h, w, ch]

        img_canvas = fill_canvas(x, n, h, w, c)
        rec_canvas = fill_canvas(recs, n, h, w, c)
        sample_canvas = fill_canvas(samples, n, h, w, c)

        return sample_canvas, rec_canvas, img_canvas

    def save(self, fp):
        self.save_weights(f"{fp}_46")

    def load(self, fp):
        self.load_weights(f"{fp}_46")

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model = f"model46-{time}"
        train_log_dir = f"/tmp/{experiment}/{model}/train"
        val_log_dir = f"/tmp/{experiment}/{model}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{experiment}/{model}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model46.py > models/model46.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model46()

    # intialize model
    model.val_batch()

    layer = (
        model.encoder.convs.layers[1]
        .blocks.layers[-1]
        .conv.layers[-1]
        .trainable_weights[0]
    )

    for encblock in model.encoder.convs.layers[1:-1]:
        end_weights = encblock.blocks.layers[-1].conv.layers[-1].trainable_weights[0]
        end_weights.assign(end_weights * tf.math.sqrt(1 / 5))

    for decblock in model.decoder.deconvs.layers[:-1]:
        end_weights = decblock.blocks.layers[-1].conv.layers[-1].trainable_weights[0]
        end_weights.assign(end_weights * tf.math.sqrt(1 / 5))

    train(model, n_updates=1_000_000, eval_interval=1000)

    model.load("best")
    mean_llh, llh = model.test(5000)

    print(mean_llh)

    # x, y = next(model.ds.train_loader)
    # model(x)
    # model.load("best")
    # qzx = model.encode(x)
    # z = qzx.sample(5)
    # print(np.std(qzx.loc), np.std(z))
    # p = model.decode(z)
    # pz = tfd.Normal(tf.zeros_like(z), tf.ones_like(z))
    # z2 = pz.sample()
    # p = model.decode(z)

"""
combination of model43 (or model45/46) and model36
model36: 2 stochastic layers.
model45: res blocks

Looks pretty good, but with some RGB sampling artefacts in the samples
TODO: try with mdl
"""
import os
from collections import namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from models.model import Model
from modules import DecoderBlock, EncoderBlock
from utils import DiscretizedLogistic, GlobalStep, logmeanexp, setup_data

# ---- collection of distribution and sample
BaseDist = namedtuple("Dist", "dist sample axes")


class Dist(namedtuple("Dist", "dist sample axes")):
    """
    Holds either variational or generative distribution data

    dist : tfd.Distribution
    samples : samples from this distribution
    axes : which axes to sum over in a loss
    """

    @property
    def z(self):
        return self.sample

    @property
    def x(self):
        return self.sample

    @property
    def p(self):
        return self.dist

    @property
    def q(self):
        return self.dist


@property
def axes(self):
    return self._axes


@axes.setter
def axes(self, axes):
    self._axes = axes


tfd.Distribution.axes = axes


def loss_fn(x, pz, qz1x, qz2z1, pz1z2, pxz1):

    lqz2z1 = tf.reduce_sum(qz2z1.q.log_prob(qz2z1.z), axis=qz2z1.axes)
    lqz1x = tf.reduce_sum(qz1x.q.log_prob(qz1x.z), axis=qz1x.axes)

    lpz2 = tf.reduce_sum(pz.log_prob(qz2z1.z), axis=pz.axes)
    lpz1z2 = tf.reduce_sum(pz1z2.p.log_prob(qz1x.z), axis=qz1x.axes)
    lpxz = tf.reduce_sum(pxz1.p.log_prob(x), axis=pxz1.axes)

    log_w = lpxz + (lpz2 - lqz2z1) + (lpz1z2 - lqz1x)

    # logmeanexp over samples, average over batch
    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

    # bits_pr_dim:
    # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146
    n_dims = tf.cast(tf.math.reduce_prod(x.shape[-len(pxz1.axes) :]), tf.float32)
    bpd = -iwae_elbo / (tf.math.log(2.0) * n_dims)

    log_snis = tf.math.log_softmax(log_w)
    kl1 = -tf.reduce_mean(lpz1z2 - lqz1x, axis=0)
    kl2 = -tf.reduce_mean(lpz2 - lqz2z1, axis=0)

    return -iwae_elbo, {
        "iwae_elbo": iwae_elbo,
        "bpd": bpd,
        "lpxz": lpxz,  # tf.reduce_logsumexp(lpxz + log_snis, axis=0),
        "lqz1x": lqz1x,
        "lqz2z1": lqz2z1,
        "lpz2": lpz2,
        "lpz1z2": lpz1z2,
        "kl1": kl1,
        "kl2": kl2,
    }


class DataSets:
    def __init__(self, ds="svhn_cropped"):
        self.train_loader, self.val_loader, self.ds_test = setup_data(ds)


class BasicBlock(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.gelu)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.gelu)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.nn.softplus)

    def call(self, inputs, n_samples=None):
        h1 = self.l1(inputs)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        p = tfd.Normal(q_mu, q_std + 1e-6)
        sample = p.sample(n_samples if n_samples is not None else [])

        return Dist(dist=p, sample=sample, axes=[-1])


class Encoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()

        n_latent = 20
        filters = 64
        hidden = 32
        in_shape = (32, 32, 3)

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
                    hidden, filters, n_blocks=3, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=3, downscale_rate=2, rezero=True
                ),
                EncoderBlock(
                    hidden, filters, n_blocks=3, downscale_rate=2, rezero=True
                ),
            ]
        )

        self.fc = layers.Dense(2 * n_latent)

    def call(self, x, n_samples=1, **kwargs):
        out = self.convs(x)
        out = tf.reshape(out, [out.shape[0], -1])
        mu, logstd = tf.split(self.fc(out), num_or_size_splits=2, axis=-1)
        q = tfd.Normal(mu, tf.nn.softplus(logstd))
        z = q.sample(n_samples)
        return Dist(q, z, axes=[-1])


class Decoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()
        self.out_shape = (32, 32, 3)

        filters = 64
        hidden = 32

        self.base_size = [
            self.out_shape[0] // 2 ** 3,
            self.out_shape[1] // 2 ** 3,
            filters,
        ]
        self.fc = layers.Dense(np.prod(self.base_size), activation=tf.nn.gelu)

        self.deconvs = tf.keras.Sequential(
            [
                DecoderBlock(hidden, filters, n_blocks=3, upscale_rate=2, rezero=True),
                DecoderBlock(hidden, filters, n_blocks=3, upscale_rate=2, rezero=True),
                DecoderBlock(hidden, filters, n_blocks=3, upscale_rate=2, rezero=True),
                layers.Conv2D(
                    3 * 2, strides=1, kernel_size=3, padding="same", activation=None
                ),
            ]
        )

    def call(self, z, **kwargs):
        h = self.fc(z)
        # ---- merge sample and batch dimensions and reshape from dense to conv, mirrored from encoder
        h = tf.reshape(h, [-1, *self.base_size])
        out = self.deconvs(h)
        out = tf.reshape(
            out, [*z.shape[:-1], self.out_shape[0], self.out_shape[1], 3 * 2]
        )
        mu, logstd = tf.split(out, num_or_size_splits=2, axis=-1)
        # pxz = DiscretizedLogistic(mu, tf.nn.tanh(logstd), low=0.0, high=1.0, levels=256)
        pxz = DiscretizedLogistic(mu, logstd, low=0.0, high=1.0, levels=256)
        x = pxz.sample()
        return Dist(pxz, x, axes=[-1, -2, -3])


class Model49(Model, tf.keras.Model):
    def __init__(self):
        super(Model49, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 5
        self.n_latent = 20
        self.global_step = GlobalStep()
        self.global_step.bind_to(
            self.update_learning_rate
        )  # add callback to update learning rate when gs.value is changed
        self.init_tensorboard()

        self.loss_fn = loss_fn

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1]

        self.encoder = Encoder(self.n_latent)
        self.mlp_encoder = BasicBlock(n_hidden=100, n_latent=self.n_latent)
        self.decoder = Decoder(self.n_latent)
        self.mlp_decoder = BasicBlock(n_hidden=100, n_latent=self.n_latent)

        self.ds = DataSets()

    def update_learning_rate(self, value):
        if value in [2 ** i * 7000 for i in range(8)]:
            old_lr = self.optimizer.learning_rate.numpy()
            new_lr = 1e-3 * 10 ** (-value / (2 ** 7 * 7000))
            self.optimizer.learning_rate.assign(new_lr)
            print(f"Changing learningrate from {old_lr:.2e} to {new_lr:.2e}")

    def encode(self, x, n_samples=1, **kwargs):
        qz1x = self.encoder(x, n_samples)
        qz2z1 = self.mlp_encoder(qz1x.z)
        return qz1x, qz2z1

    def decode(self, z1, z2, **kwargs):
        pz1z2 = self.mlp_decoder(z2)
        pxz1 = self.decoder(z1)
        return pz1z2, pxz1

    def generate(self, z2, **kwargs):
        pz1z2 = self.mlp_decoder(z2)
        pxz1 = self.decoder(pz1z2.z)
        return pz1z2, pxz1

    def call(self, x, n_samples=1, **kwargs):
        qz1x, qz2z1 = self.encode(x, n_samples)
        pz1z2, pxz1 = self.decode(qz1x.z, qz2z1.z)
        return qz1x, qz2z1, pz1z2, pxz1

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            qz1x, qz2z1, pz1z2, pxz1 = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(x, self.pz, qz1x, qz2z1, pz1z2, pxz1)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, x):
        qz1x, qz2z1, pz1z2, pxz1 = self(x, n_samples=self.n_samples)
        loss, metrics = self.loss_fn(x, self.pz, qz1x, qz2z1, pz1z2, pxz1)
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
            qz1x, qz2z1, pz1z2, pxz1 = self(x[None, :], n_samples=n_samples)
            loss, metrics = self.loss_fn(x, self.pz, qz1x, qz2z1, pz1z2, pxz1)
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
        qz1x, qz2z1, pz1z2, pxz1 = self(x[: n ** 2], n_samples=1)
        recs = pxz1.p.mean()[0]  # [n_samples, batch, h, w, ch]

        rec_canvas = np.empty([n * h, n * w, c])
        for i in range(n):
            for j in range(n):
                rec_canvas[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = recs[
                    i * n + j, :, :, :
                ]

        pz = tfd.Normal(tf.zeros_like(qz2z1.z), tf.ones_like(qz2z1.z))
        pz1z2, pxz1 = self.generate(pz.sample())
        # samples = np.clip(pxz1.p.mean()[0], 0.0, 1.0)  # [n_samples, batch, h, w, ch]
        samples = np.clip(pxz1.p.sample()[0], 0.0, 1.0)  # [n_samples, batch, h, w, ch]

        sample_canvas = np.empty([n * h, n * w, c])
        for i in range(n):
            for j in range(n):
                sample_canvas[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = samples[
                    i * n + j, :, :, :
                ]

            img_canvas = np.empty([n * h, n * w, c])
            for i in range(n):
                for j in range(n):
                    img_canvas[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = x[
                        i * n + j, :, :, :
                    ]

        return sample_canvas, rec_canvas, img_canvas

    def save(self, fp):
        self.save_weights(f"{fp}_49")

    def load(self, fp):
        self.load_weights(f"{fp}_49")

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model = f"model49-{time}"
        train_log_dir = f"/tmp/{experiment}/{model}/train"
        val_log_dir = f"/tmp/{experiment}/{model}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{experiment}/{model}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model49.py > models/model49.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model49()

    # intialize model
    model.val_batch()

    train(model, n_updates=1_000_000, eval_interval=1000)

    model.load("best")
    mean_llh, llh = model.test(1000)

    print(mean_llh)

    # x, y = next(model.ds.train_loader)
    # model(x)
    # model.load("best")
    # qzx = model.encode(x)
    # z = qzx.sample(model.n_samples)
    # pxz = model.decode(z)
    #
    # ones = tf.ones_like(z)
    #
    # for i in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    #     pxz = model.decode(i * ones)
    #     print(f" |z|: {i}, pxz.scale: {np.mean(pxz.logscale):.4f}")
    #     # print(np.std(pxz.loc))
    #
    # for i in [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
    #     pxz = model.decode(i * ones)
    #     print(f" |z|: {i}, pxz.scale: {np.mean(pxz.logscale):.4f}")
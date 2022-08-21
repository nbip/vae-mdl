"""
Two stochastic layers.
The IWAE loss with two stochastic layers is spelled out for clarity.
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from models.model import Model
from utils import (
    DiscretizedLogistic,
    DistributionTuple,
    GlobalStep,
    fill_canvas,
    logmeanexp,
    setup_data,
)


@property
def axes(self):
    return self._axes


@axes.setter
def axes(self, axes):
    self._axes = axes


tfd.Distribution.axes = axes


def loss_fn(x, pz, qz1x, qz2z1, pz1z2, pxz1):

    lqz2z1 = tf.reduce_sum(qz2z1.dist.log_prob(qz2z1.z), axis=qz2z1.axes)
    lqz1x = tf.reduce_sum(qz1x.dist.log_prob(qz1x.z), axis=qz1x.axes)

    lpz2 = tf.reduce_sum(pz.log_prob(qz2z1.z), axis=pz.axes)
    lpz1z2 = tf.reduce_sum(pz1z2.dist.log_prob(qz1x.z), axis=qz1x.axes)
    lpxz = tf.reduce_sum(pxz1.dist.log_prob(x), axis=pxz1.axes)

    log_w = lpxz + (lpz2 - lqz2z1) + (lpz1z2 - lqz1x)

    # ---- logmeanexp over samples, average over batch
    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

    # ---- bits_pr_dim:
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

        return DistributionTuple(dist=p, sample=sample, axes=(-1,))


class Encoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()

        self.convs = tf.keras.Sequential(
            [
                layers.Conv2D(
                    32, strides=1, kernel_size=3, padding="same", activation=tf.nn.gelu
                ),
                layers.Conv2D(
                    64, strides=2, kernel_size=3, padding="same", activation=tf.nn.gelu
                ),
                layers.Conv2D(
                    128, strides=2, kernel_size=3, padding="same", activation=tf.nn.gelu
                ),
                layers.Conv2D(
                    256, strides=2, kernel_size=3, padding="same", activation=tf.nn.gelu
                ),
            ]
        )

        conv_out_dim = 32 // 2 ** 3 * 32 // 2 ** 3 * 256
        self.fc = layers.Dense(2 * n_latent)

    def call(self, x, n_samples=1, **kwargs):
        out = self.convs(x)
        out = tf.reshape(out, [out.shape[0], -1])
        mu, logstd = tf.split(self.fc(out), num_or_size_splits=2, axis=-1)
        q = tfd.Normal(mu, tf.nn.softplus(logstd))
        z = q.sample(n_samples)
        return DistributionTuple(q, z, axes=(-1,))


class Decoder(tf.keras.Model):
    def __init__(self, n_latent):
        super().__init__()
        out_shape = (32, 32, 3)

        self.base_size = [out_shape[0] // 2 ** 3, out_shape[1] // 2 ** 3, 128]
        self.fc = layers.Dense(np.prod(self.base_size), activation=tf.nn.gelu)

        self.deconvs = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(
                    128, kernel_size=4, strides=2, padding="same", activation=tf.nn.gelu
                ),
                layers.Conv2DTranspose(
                    64, kernel_size=4, strides=2, padding="same", activation=tf.nn.gelu
                ),
                layers.Conv2DTranspose(
                    32, kernel_size=4, strides=2, padding="same", activation=tf.nn.gelu
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
        # pxz = DiscretizedLogistic(mu, tf.nn.tanh(logstd), low=0.0, high=1.0, levels=256)
        pxz = DiscretizedLogistic(mu, logstd, low=0.0, high=1.0, levels=256)
        x = pxz.sample()
        return DistributionTuple(pxz, x, axes=(-1, -2, -3))


class Model06(Model, tf.keras.Model):
    def __init__(self):
        super(Model06, self).__init__()

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

        # reconstructions
        qz1x, qz2z1, pz1z2, pxz1 = self(x[: n ** 2], n_samples=1)
        recs = pxz1.dist.mean()[0]  # [n_samples, batch, h, w, ch]

        # samples
        pz = tfd.Normal(tf.zeros_like(qz2z1.z), tf.ones_like(qz2z1.z))
        pz1z2, pxz1 = self.generate(pz.sample())
        # samples = np.clip(pxz1.p.mean()[0], 0.0, 1.0)  # [n_samples, batch, h, w, ch]
        samples = np.clip(pxz1.dist.sample()[0], 0.0, 1.0)

        img_canvas = fill_canvas(x, n, h, w, c)
        rec_canvas = fill_canvas(recs, n, h, w, c)
        sample_canvas = fill_canvas(samples, n, h, w, c)

        return sample_canvas, rec_canvas, img_canvas

    def save(self, fp):
        self.save_weights(f"{self.save_dir}/{fp}")

    def load(self, fp):
        self.load_weights(f"{self.save_dir}/{fp}")

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model = f"model06"
        train_log_dir = f"/tmp/{experiment}/{model}-{time}/train"
        val_log_dir = f"/tmp/{experiment}/{model}-{time}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{model}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model06.py > models/model06.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model06()

    # intialize model
    model.val_batch()
    model.val_batch()

    train(model, n_updates=100_000, eval_interval=1000)

    model.load("best")
    mean_llh, llh = model.test(5000)

    print(mean_llh)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = next(model.ds.train_loader)
    samples, recs, imgs = model._plot_samples(x)

    plt.clf()
    plt.imshow(samples)
    plt.axis("off")
    plt.savefig(f"./assets/model06_samples")

    plt.clf()
    plt.imshow(recs)
    plt.axis("off")
    plt.savefig(f"./assets/model06_recs")

    plt.clf()
    plt.imshow(imgs)
    plt.axis("off")
    plt.savefig(f"./assets/model06_imgs")

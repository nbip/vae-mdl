"""
Stochastic blocks all the way
Rexonstructions are good but samples are awful.
TODO: figure out a nice encoding/decoding scheme.
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
from modules import (
    DecoderBlock,
    EncoderBlock,
    StochasticDecoderBlock,
    StochasticEncoderBlock,
)
from utils import DiscretizedLogistic, GlobalStep, fill_canvas, logmeanexp, setup_data


class DataSets:
    def __init__(self):
        self.train_loader, self.val_loader, self.ds_test = setup_data(
            "svhn_cropped", batch_size=32, val_batch_size=32
        )


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        filters = 64
        hidden = 32
        n_blocks = 5
        in_shape = (32, 32, 3)

        self.conv = layers.Conv2D(
            filters, strides=1, kernel_size=3, padding="same", activation=tf.nn.gelu
        )

        self.layer_list = [
            StochasticEncoderBlock(
                hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
            ),
            StochasticEncoderBlock(
                hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
            ),
            StochasticEncoderBlock(
                hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
            ),
            StochasticEncoderBlock(
                hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
            ),
            StochasticEncoderBlock(
                hidden, filters, n_blocks=n_blocks, downscale_rate=2, rezero=True
            ),
        ]

    def call(self, x, n_samples=1, **kwargs):
        x = self.conv(x)

        # add sample dimension and repeat n_samples
        z = tf.repeat(x[None], repeats=n_samples, axis=0)

        Qs, Zs = [], []
        for layer in self.layer_list:
            q = layer(z)
            z = q.sample()
            Qs.append(q)
            Zs.append(z)

        return Qs, Zs


class PLayer(layers.Layer):
    def __init__(self):
        super().__init__()

        filters = 64
        hidden = 32
        n_blocks = 5

        self.conv = tf.keras.Sequential(
            [
                DecoderBlock(
                    hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
                ),
                layers.Conv2D(
                    2 * 3,
                    strides=1,
                    kernel_size=3,
                    padding="same",
                    activation=None,
                ),
            ]
        )

    def call(self, x):
        mu, logstd = tf.split(self.conv(x), num_or_size_splits=2, axis=-1)
        return DiscretizedLogistic(mu, logstd, low=0.0, high=1.0, levels=256)


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out_shape = (32, 32, 3)
        filters = 64
        hidden = 32
        n_blocks = 5

        self.layer_list = [
            StochasticDecoderBlock(
                hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
            ),
            StochasticDecoderBlock(
                hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
            ),
            StochasticDecoderBlock(
                hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
            ),
            StochasticDecoderBlock(
                hidden, filters, n_blocks=n_blocks, upscale_rate=2, rezero=True
            ),
            PLayer(),
        ]

    def call(self, Zs, **kwargs):

        Ps = []
        for z, layer in zip(reversed(Zs), self.layer_list):
            Ps.append(layer(z))

        return Ps

    def sample(self, z):

        for layer in self.layer_list:
            p = layer(z)
            z = p.sample()

        return p, z


class Model47(Model, tf.keras.Model):
    def __init__(self):
        super(Model47, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 5
        self.global_step = GlobalStep()
        self.global_step.bind_to(
            self.update_learning_rate
        )  # add callback to update learning rate when gs.value is changed
        self.init_tensorboard()

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

    def loss_fn(self, x, zs, qs, ps, prior):
        """
        :param x: image data [b, h, w, c]
        :param zs: list of latent samples from each stochastic layer
        :param qs: list of variational posterior distributions
        :param ps: list of generative distributions
        :param prior: top stochastic layer prior
        :return: -elbo, metrics
        """

        # ---- prior p(z)
        lprior = tf.reduce_sum(prior.log_prob(zs[-1]), axis=prior.axes)

        # ---- rest of p(z_l | z_{l-1})
        assert len(zs[:-1]) == len(ps[:-1])
        lpz = [
            tf.reduce_sum(p.log_prob(z), axis=[-1, -2, -3])
            for p, z in zip(ps[:-1], reversed(zs[:-1]))
        ]

        lpxz = tf.reduce_sum(ps[-1].log_prob(x), axis=[-1, -2, -3])

        lqz = [tf.reduce_sum(q.log_prob(z), axis=[-1, -2, -3]) for z, q in zip(zs, qs)]

        log_w = lpxz + tf.add_n(lpz + [lprior]) - tf.add_n(lqz)

        # logmeanexp over samples, average over batch
        iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

        # bits_pr_dim:
        # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
        # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146
        n_dims = tf.cast(tf.math.reduce_prod(x.shape[1:]), tf.float32)
        bpd = -iwae_elbo / (tf.math.log(2.0) * n_dims)

        return -iwae_elbo, {
            "iwae_elbo": iwae_elbo,
            "bpd": bpd,
            "lpxz": lpxz,
        }

    def call(self, x, n_samples=1, **kwargs):
        q, z = self.encoder(x, n_samples)
        p = self.decoder(z)
        return q, z, p

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            q, z, p = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(x, z, q, p, self.pz)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, x):
        q, z, p = self(x, n_samples=self.n_samples)
        loss, metrics = self.loss_fn(x, z, q, p, self.pz)
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
        n = np.min([n, int(np.sqrt(x.shape[0]))])

        # reconstructions
        qs, zs, ps = self(x[: n ** 2], n_samples=1)
        recs = ps[-1].mean()[0]  # [n_samples, batch, h, w, ch]

        # samples
        pz = tfd.Normal(tf.zeros_like(zs[-1]), tf.ones_like(zs[-1]))
        pxz, sample = self.decoder.sample(pz.sample())
        # TODO: use real samples here
        samples = np.clip(pxz.mean()[0], 0.0, 1.0)  # [n_samples, batch, h, w, ch]

        img_canvas = fill_canvas(x, n, h, w, c)
        rec_canvas = fill_canvas(recs, n, h, w, c)
        sample_canvas = fill_canvas(samples, n, h, w, c)

        return sample_canvas, rec_canvas, img_canvas

    def save(self, fp):
        self.save_weights(f"{fp}_47")

    def load(self, fp):
        self.load_weights(f"{fp}_47")

    def init_tensorboard(self, name: str = None) -> None:
        experiment = name or "tensorboard"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model = f"model47-{time}"
        train_log_dir = f"/tmp/{experiment}/{model}/train"
        val_log_dir = f"/tmp/{experiment}/{model}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{experiment}/{model}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model47.py > models/model47.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model47()

    # intialize model
    model.val_batch()

    # for encblock in model.encoder.convs.layers[1:-1]:
    #     end_weights = encblock.blocks.layers[-1].conv.layers[-1].trainable_weights[0]
    #     end_weights.assign(end_weights * tf.math.sqrt(1 / 5))
    #
    # for decblock in model.decoder.deconvs.layers[:-1]:
    #     end_weights = decblock.blocks.layers[-1].conv.layers[-1].trainable_weights[0]
    #     end_weights.assign(end_weights * tf.math.sqrt(1 / 5))

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
    # print(np.min(p.loc), np.max(p.loc))
    # pz = tfd.Normal(tf.zeros_like(z), tf.ones_like(z))
    # z2 = pz.sample()
    # p2 = model.decode(z2)
    # print(np.min(p2.loc), np.max(p2.loc))
    #
    # # --- check
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # ds = DataSets()
    # x, y = next(ds.train_loader)
    #
    # enc = Encoder()
    # dec = Decoder()
    #
    # q, z = enc(x)
    # p = dec(z)

"""
Two stochastic layers, residual blocks in encoder/decoder,
plain discretized logistic loss

looks like model51, but is nan prone! introdice gradient clipping

TODO: The almost similar model49 is not as nan-prone, why?

TODO: take the decoder and analyze the receptive field on a given z...

TODO: make global_step a tf.Variable so it is loaded with the model: https://stackoverflow.com/a/36117853

cues:
VDVAE:
https://github.com/openai/vdvae/blob/main/train.py#L17
https://github.com/openai/vdvae/blob/main/vae.py#L238
https://github.com/openai/vdvae/blob/main/vae.py#L231
https://github.com/openai/vdvae/blob/main/vae.py#L197
https://github.com/openai/vdvae/blob/main/vae.py#L156
https://github.com/openai/vdvae/blob/main/vae.py#L150

simple VDVAE:
https://github.com/vvvm23/vdvae/blob/main/vae.py#L208
https://github.com/vvvm23/vdvae/blob/main/vae.py#L112

Code tips:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L534
this is how my API should be:
https://github.com/lucidrains/denoising-diffusion-pytorch 
"""
import os
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from models.model import Model
from modules import DecoderBlock, EncoderBlock
from utils import (
    DiscretizedLogistic,
    DistributionTuple,
    GlobalStep,
    fill_canvas,
    logmeanexp,
    setup_data,
)


def loss_fn(
    x: tf.Tensor,
    Qs: Dict[int, DistributionTuple],
    Ps: Dict[int, DistributionTuple],
    pxz: tfp.distributions.Distribution,
    prior: DistributionTuple,
):

    top_layer = max(Qs.keys())
    KL = {}

    # ---- prior p(z)
    p, _, paxes = list(prior)
    q, z, qaxes = list(Qs[top_layer])
    log_p = tf.reduce_sum(p.log_prob(z), axis=paxes)
    log_q = tf.reduce_sum(q.log_prob(z), axis=qaxes)
    KL[f"kl{top_layer}"] = log_p - log_q

    # ---- stochastic layers 1 : L-1
    for i in range(1, top_layer):
        q, z, qaxes = list(Qs[i])
        p, _, paxes = list(Ps[i])

        log_q = tf.reduce_sum(q.log_prob(z), axis=qaxes)
        log_p = tf.reduce_sum(p.log_prob(z), axis=paxes)
        KL[f"kl{i}"] = log_p - log_q

    # ---- observation model p(x | z_1)
    lpxz = tf.reduce_sum(pxz.dist.log_prob(x), axis=pxz.axes)

    # ---- log weights
    log_w = lpxz + tf.add_n(list(KL.values()))

    # ---- logmeanexp over samples, average over batch
    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

    # ---- bits_pr_dim:
    # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146
    n_dims = tf.cast(tf.math.reduce_prod(x.shape[1:]), tf.float32)
    bpd = -iwae_elbo / (tf.math.log(2.0) * n_dims)

    return -iwae_elbo, {"iwae_elbo": iwae_elbo, "bpd": bpd, "lpxz": lpxz, **KL}


class DataSets:
    def __init__(self, ds: str = "svhn_cropped") -> None:
        self.train_loader, self.val_loader, self.ds_test = setup_data(ds)


class BasicBlock(tf.keras.Model):
    def __init__(self, n_hidden: int, n_latent: int, **kwargs) -> None:
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.gelu)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.gelu)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.nn.softplus)

    def call(
        self, inputs: tf.Tensor, n_samples: Optional[int] = None
    ) -> DistributionTuple:
        h1 = self.l1(inputs)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        p = tfd.Normal(q_mu, q_std + 1e-6)
        sample = p.sample(n_samples if n_samples is not None else [])

        return DistributionTuple(dist=p, sample=sample, axes=(-1,))


class Encoder(tf.keras.Model):
    def __init__(self, n_latent: int):
        super().__init__()

        filters = 64
        hidden = 32

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

    def call(self, x: tf.Tensor, n_samples: int = 1, **kwargs) -> DistributionTuple:
        out = self.convs(x)
        out = tf.reshape(out, [out.shape[0], -1])
        mu, logstd = tf.split(self.fc(out), num_or_size_splits=2, axis=-1)
        q = tfd.Normal(mu, tf.nn.softplus(logstd) + 1e-6)
        z = q.sample(n_samples)
        return DistributionTuple(q, z, axes=(-1,))


class Decoder(tf.keras.Model):
    def __init__(self):
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

    def call(self, z: tf.Tensor, **kwargs) -> DistributionTuple:
        h = self.fc(z)
        # ---- merge sample and batch dimensions and reshape from dense to conv, mirrored from encoder
        h = tf.reshape(h, [-1, *self.base_size])
        out = self.deconvs(h)
        out = tf.reshape(out, [*z.shape[:-1], 32, 32, 3 * 2])
        mu, logstd = tf.split(out, num_or_size_splits=2, axis=-1)
        pxz = DiscretizedLogistic(mu, logstd, low=0.0, high=1.0, levels=256)
        x = pxz.sample()
        return DistributionTuple(pxz, x, axes=(-1, -2, -3))


class Model53(Model, tf.keras.Model):
    def __init__(self):
        super(Model53, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.n_samples = 5
        self.n_latent = 20
        self.global_step = GlobalStep()
        self.global_step.bind_to(
            self.update_learning_rate
        )  # add callback to update learning rate when gs.value is changed
        self.init_tensorboard()

        self.loss_fn = loss_fn

        self.pz = DistributionTuple(tfd.Normal(0.0, 1.0), None, (-1,))

        self.encoder = Encoder(self.n_latent)
        self.mlp_encoder = BasicBlock(n_hidden=100, n_latent=self.n_latent)
        self.decoder = Decoder()
        self.mlp_decoder = BasicBlock(n_hidden=100, n_latent=self.n_latent)

        self.ds = DataSets()

    def update_learning_rate(self, value: int) -> None:
        if value in [2 ** i * 7000 for i in range(8)]:
            old_lr = self.optimizer.learning_rate.numpy()
            new_lr = 1e-3 * 10 ** (-value / (2 ** 7 * 7000))
            self.optimizer.learning_rate.assign(new_lr)
            print(f"Changing learningrate from {old_lr:.2e} to {new_lr:.2e}")

    def encode(
        self, x: tf.Tensor, n_samples: int = 1, **kwargs
    ) -> Dict[int, DistributionTuple]:
        Qs = {}

        q1 = self.encoder(x, n_samples)
        Qs[1] = q1

        q2 = self.mlp_encoder(q1.z)
        Qs[2] = q2
        return Qs

    def decode(
        self, Qs: Dict[int, DistributionTuple], **kwargs
    ) -> Tuple[Dict[int, DistributionTuple], DistributionTuple]:

        Ps = {}
        q2 = Qs[2]
        q1 = Qs[1]

        p1 = self.mlp_decoder(q2.z)
        Ps[1] = p1

        pxz = self.decoder(q1.z)
        return Ps, pxz

    def generate(
        self, z: Optional[tf.Tensor] = None, **kwargs
    ) -> Tuple[DistributionTuple, DistributionTuple]:
        pz1z2 = self.mlp_decoder(z)
        pxz1 = self.decoder(pz1z2.z)
        return pz1z2, pxz1

    def call(
        self, x: tf.Tensor, n_samples: int = 1, **kwargs
    ) -> Tuple[
        Dict[int, DistributionTuple], Dict[int, DistributionTuple], DistributionTuple
    ]:
        Qs = self.encode(x, n_samples)
        Ps, pxz = self.decode(Qs)
        return Qs, Ps, pxz

    @tf.function
    def train_step(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        with tf.GradientTape() as tape:
            Qs, Ps, pxz = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(x, Qs, Ps, pxz, self.pz)

        grads = tape.gradient(loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 5.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        Qs, Ps, pxz = self(x, n_samples=self.n_samples)
        loss, metrics = self.loss_fn(x, Qs, Ps, pxz, self.pz)
        return loss, metrics

    def train_batch(self) -> Tuple[tf.Tensor, Dict]:
        x, y = next(self.ds.train_loader)
        loss, metrics = self.train_step(x)
        self.global_step.value += 1
        return loss, metrics

    def val_batch(self) -> Tuple[tf.Tensor, Dict]:
        x, y = next(self.ds.val_loader)
        loss, metrics = self.val_step(x)
        self.report(x, metrics)
        return loss, metrics

    def test(self, n_samples: int) -> Tuple[float, List]:
        llh = np.nan * np.zeros(len(self.ds.ds_test))

        for i, (x, y) in enumerate(tqdm(self.ds.ds_test)):
            Qs, Ps, pxz = self(x[None, :], n_samples=n_samples)
            loss, metrics = self.loss_fn(x, Qs, Ps, pxz, self.pz)
            llh[i] = metrics["iwae_elbo"]

        return llh.mean(), llh

    def report(self, x: tf.Tensor, metrics: Dict):
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

    def _plot_samples(self, x: tf.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, h, w, c = 8, 32, 32, 3

        # reconstructions
        Qs, Ps, pxz = self(x[: n ** 2], n_samples=1)
        recs = pxz.dist.mean()[0]  # [n_samples, batch, h, w, ch]

        # samples
        top_layer = max(Qs.keys())
        pz = tfd.Normal(tf.zeros_like(Qs[top_layer].z), tf.ones_like(Qs[top_layer].z))
        pz1z2, pxz1 = self.generate(pz.sample())
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
        model = f"model53"
        train_log_dir = f"/tmp/{experiment}/{model}-{time}/train"
        val_log_dir = f"/tmp/{experiment}/{model}-{time}/val"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # ---- directory for saving trained models
        self.save_dir = f"./saved_models/{model}"
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u models/model53.py > models/model53.log &
    from trainer import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = Model53()

    # intialize model
    model.val_batch()

    # for i in range(1000):
    #     res = model.train_batch()
    #     print(f"{i}: {res[0]:.2f}")

    train(model, n_updates=1_000_000, eval_interval=1000)

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
    plt.savefig(f"./assets/model53_samples")

    plt.clf()
    plt.imshow(recs)
    plt.axis("off")
    plt.savefig(f"./assets/model53_recs")

    plt.clf()
    plt.imshow(imgs)
    plt.axis("off")
    plt.savefig(f"./assets/model53_imgs")

    # debug
    x, y = next(model.ds.train_loader)
    Qs, Ps, pxz = model(x, n_samples=model.n_samples)
    loss, metrics = model.loss_fn(x, Qs, Ps, pxz, model.pz)

    # ---- reload and restart training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = Model53()

    # intialize model
    model.val_batch()

    # load best
    model.load("best")
    # restoring global_step: https://stackoverflow.com/a/36117853

    train(model, n_updates=1_000_000, eval_interval=1000)

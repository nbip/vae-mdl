"""
Same as model03.py, but with a dense connection to the latent space

https://github.com/rasmusbergpalm/vnca/blob/dmg-double-celeba/vae-nca.py
https://github.com/rasmusbergpalm/vnca/tree/dmg_celebA_baseline
https://github.com/rasmusbergpalm/vnca/blob/dmg_celebA_baseline/modules/vae.py#L88

TODO: run VAE ELBO and beta-VAE elbo https://github.com/rasmusbergpalm/vnca/blob/dmg-double-celeba/vae-nca.py#L282
TODO: use the torch celeba loader, adapt to tensorflow
TODO: change module load in .bashrc to match the tf3 environment

# TODO: make the most minimal change to dml: instead of x use loc, so
# loc_g = loc_g + coeff * clip(loc_r, -1, 1)
# I think that is the first incremental change I can make

# TODO: use architecture from supMIWAE (which worked decently on SVHN)

Things that mitigated nans:
- softplus instead of exp
- kernel initializer, look to https://arxiv.org/pdf/2203.13751.pdf appendix C.1
"""

import os
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from models.loss import iwae_loss
from models.model import Model
from utils import MixtureDiscretizedLogistic


class GatedBlock(layers.Layer):
    """https://arxiv.org/pdf/1612.08083.pdf"""

    def __init__(self, filters=64, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        self.l1 = layers.Conv2D(
            filters, kernel_size=3, strides=1, padding="same", activation=activation
        )
        self.l2 = layers.Conv2D(
            2 * filters, kernel_size=3, strides=1, padding="same", activation=activation
        )

    def call(self, inputs, **kwargs):
        block_input = self.l1(inputs)
        A, B = tf.split(self.l2(block_input), 2, axis=-1)
        H = A * tf.nn.sigmoid(B)
        return H + block_input


class Encoder(tf.keras.Model):
    def __init__(self, n_latent=20, **kwargs):
        super().__init__(**kwargs)

        kernel_size = 4
        activation = tf.nn.relu

        self.c1 = tf.keras.Sequential(
            [
                layers.Conv2D(
                    32,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    activation=activation,
                ),
                layers.Conv2D(
                    64,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    activation=activation,
                ),
                layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", activation=activation
                ),
                *[GatedBlock(64) for _ in range(4)],
            ]
        )

        # ---- dense
        self.dense = layers.Dense(n_latent * 2)

    def call(self, x, **kwargs):
        shape = x.shape
        h = self.c1(x)
        h = tf.reshape(h, [*shape[:-3], -1])  # flatten
        return self.dense(h)


class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kernel_size = 4
        activation = tf.nn.relu

        self.dense = layers.Dense(64 * 8 * 8, activation=activation)

        self.c1 = tf.keras.Sequential(
            [
                layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", activation=activation
                ),
                *[GatedBlock(64) for _ in range(4)],
                layers.Conv2DTranspose(
                    64,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    activation=activation,
                ),
                layers.Conv2DTranspose(
                    32,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    activation=activation,
                ),
                layers.Conv2DTranspose(100, kernel_size=3, strides=1, padding="same"),
            ]
        )

    def call(self, z, **kwargs):
        shape = z.shape  # [samples, batch, dims] or [batch, dims]
        h = self.dense(z)
        # ---- merge sample and batch dimensions and reshape from dense to conv, mirrored from encoder
        h = tf.reshape(h, [-1, 8, 8, 64])
        parameters = self.c1(h)
        shape2 = parameters.shape  # [(samples) x batch, h, w, nmix * 10]

        # ---- unmerge sample and batch reshape
        parameters = tf.reshape(
            parameters, [*shape[:-1], *shape2[-3:]]
        )  # [(samples), batch, h, w, nmix * 10]

        return parameters


class Model04(Model, tf.keras.Model):
    def __init__(self):
        super(Model04, self).__init__()

        self.optimizer = tf.keras.optimizers.Adamax(1e-3)
        self.n_samples = 10
        self.global_step = 0
        self.init_tensorboard()

        self.loss_fn = iwae_loss

        self.train_loader, self.val_loader, self.ds_test = self.setup_data()

        self.pz = tfd.Normal(0.0, 1.0)
        self.pz.axes = [-1]

        # TODO: encoder/decoder: https://github.com/rasmusbergpalm/vnca/blob/dmg_celebA_baseline/baseline_celebA.py#L25
        # TODO: https://github.com/nbip/VAE-natural-images/blob/main/models/hvae.py
        # https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
        self.encoder = Encoder()
        self.decoder = Decoder()

    # TODO: debug, why can't I use tf.function here?
    # Looks like it has to return tensors
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
        p = MixtureDiscretizedLogistic(logits)
        p.axes = [-1, -2, -3]  # specify axes to sum over in log_prob
        return p

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, qzx, pxz = self(x, n_samples=self.n_samples)
            loss, metrics = self.loss_fn(x, z, self.pz, qzx, pxz)

        grads = tape.gradient(loss, self.trainable_weights)
        # grads = [tf.clip_by_norm(g, clip_norm=10.0) for g in grads]
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
        samples = pxz.sample()  # [n_samples, batch, h, w, ch]

        return samples, recs

    def setup_data(self, data_dir=None):
        def normalize(img, label):
            return tf.cast((img), tf.float32) / 255.0, label

        batch_size = 128
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            # "cifar10",
            "svhn_cropped",
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

    def save(self, fp):
        self.save_weights(f"{fp}_04")

    def load(self, fp):
        self.load_weights(f"{fp}_04")

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

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # import matplotlib;
    # matplotlib.use("Agg")  # needed when running from commandline
    import matplotlib.pyplot as plt
    import numpy as np

    b, h, w, c = 5, 32, 32, 3
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    model = Model04()

    x, y = next(model.train_loader)

    fig, ax = plt.subplots()
    ax.imshow(x[0])
    plt.show()
    plt.savefig("img")
    plt.close()

    # ---- test model subparts
    qzx = model.encode(x)
    z = qzx.sample(model.n_samples)
    pxz = model.decode(z)
    z, qzx, pxz = model(x, model.n_samples)

    # ---- test save / load
    # model.save() does not work because methods are returning tfd distribution objects
    # instead use save_weights
    # model.save_weights('saved_weights')
    # model.load_weights('saved_weights')
    # model.load_weights("best")

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

    # ---- test reporting
    model.report(x, {"loss": tf.ones(2)})

    model.train_step(x)

    model.train_batch()
    model.val_batch()

    # for i in range(100_000):
    #     loss, metrics = model.train_batch()
    #     if i % 100 == 0:
    #         print(i, loss)

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
    plt.savefig("img_rec")
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
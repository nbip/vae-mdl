"""
I had a problem using tensorflow datasets. I wnated to
- define a data-loader as an attribute of a tf.keras.Model
- use tf.random.uniform as part of preprocessing in .map()
- save model weights using model.save_weights
This caused a bunch of issues, see (not so) minimal example below

https://www.tensorflow.org/tutorials/load_data/tfrecord:
"The mapped function must operate in TensorFlow graph mode—it must operate on and return tf.Tensors. A non-tensor function, like serialize_example, can be wrapped with tf.py_function to make it compatible.
Using tf.py_function requires to specify the shape and type information that is otherwise unavailable:"

dataloader resources:
https://stackoverflow.com/a/50453698
https://stackoverflow.com/a/49916221
https://www.tensorflow.org/guide/data#randomly_shuffling_input_data
https://www.tensorflow.org/datasets/overview
https://www.tensorflow.org/datasets/splits
manual dataset https://www.tensorflow.org/datasets/add_dataset
https://www.reddit.com/r/MachineLearning/comments/65me2d/d_proper_crop_for_celeba/
celeba hint: https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/train.py#L121
download celeba: https://github.com/AntixK/PyTorch-VAE
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


class BasicBlock(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

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
                tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh),
                tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh),
                tf.keras.layers.Dense(784, activation=None),
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

        self.train_loader, self.val_loader, self.ds_test = self.setup_data()

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
        self.save_weights(f"{fp}_11____")

    def load(self, fp):
        self.load_weights(f"{fp}_11____")

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
        """model.save_weights() works with this method"""

        def normalize(img, label):
            prob = tf.cast(img, tf.float32) / 255.0
            img = tf.cast(
                tf.math.greater(
                    prob, tf.random.stateless_uniform(prob.shape, seed=(4, 2))
                ),
                tf.float32,
            )
            return img, label

        batch_size = 128
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist",
            split=["train", "test", "test"],
            shuffle_files=True,
            data_dir=data_dir,
            with_info=True,
            as_supervised=True,
        )

        Xtrain, ytrain = next(iter(ds_train.batch(len(ds_train))))
        Xval, yval = next(iter(ds_val.batch(len(ds_val))))

        Xtrain, ytrain = normalize(Xtrain, ytrain)
        Xval, yval = normalize(Xval, yval)

        ds_train = (
            tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
            .shuffle(60000)
            .repeat()
            .batch(batch_size)
            .prefetch(4)
        )
        ds_val = (
            tf.data.Dataset.from_tensor_slices((Xval, yval))
            .repeat()
            .batch(10000)
            .prefetch(4)
        )

        return iter(ds_train), iter(ds_val), ds_val

    @staticmethod
    def setup_data(data_dir=None):
        """model.save_weights() does not work with this method"""

        # tf.random.uniform cannot be used because it is not stateless and
        # for some reason model.save_weights cannot handle that
        # tf.random.uniform_stateless causes significant changes to the loss, I cannot figure out why
        # numpy approach seems to introduce some new unexpected behavior
        # related to this: https://stackoverflow.com/q/69108284
        # The Dataset is executed once, which means that the same random number is used every time... ish
        def normalize(img, label):
            prob = (
                tf.cast(img, tf.float32) / 255.0
            )  # crazy hard to debug error here if you don't cast the image first
            # img = tf.cast(tf.math.greater(prob, tf.random.uniform(prob.shape)), tf.float32)
            img = tf.cast(
                tf.math.greater(prob, tfd.Uniform().sample(prob.shape)), tf.float32
            )
            # img = tf.cast(tf.math.greater(prob, tf.random.stateless_uniform(prob.shape, seed=(4,2))), tf.float32)
            # img = tf.cast(tf.math.greater(prob, np.random.rand(*prob.shape)), tf.float32)
            # img = tf.cast(tf.math.greater(prob, 0.5), tf.float32)
            return img, label

        batch_size = 128
        data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
        os.makedirs(data_dir, exist_ok=True)

        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist",
            split=["train", "test", "test"],
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
            .batch(10000)
            .prefetch(4)
        )

        ds_test = ds_test.map(normalize).prefetch(4)

        return iter(ds_train), iter(ds_val), ds_test

    # @staticmethod
    # def setup_data(data_dir=None):
    #     batch_size = 128
    #     data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    #     os.makedirs(data_dir, exist_ok=True)
    #
    #     # ---- load data
    #     (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
    #     Ntrain = Xtrain.shape[0]
    #     Ntest = Xtest.shape[0]
    #
    #     Xtrain = Xtrain[..., None] / 255.0
    #     Xtest = Xtest[..., None] / 255.0
    #
    #     Xtrain_binarized = utils.bernoullisample(Xtrain)
    #     Xtest_binarized = utils.bernoullisample(Xtest)
    #
    #     ds_train = (
    #         tf.data.Dataset.from_tensor_slices((Xtrain_binarized, ytrain))
    #         .shuffle(Ntrain)
    #         .repeat()
    #         .batch(batch_size)
    #         .prefetch(4)
    #     )
    #     ds_test = (
    #         tf.data.Dataset.from_tensor_slices((Xtest_binarized, ytest))
    #         .repeat()
    #         .batch(Ntest)
    #         .prefetch(4)
    #     )
    #
    #     return iter(ds_train), iter(ds_test), ds_test

    # @staticmethod
    # def setup_data(data_dir=None):
    #
    #     # tf.random.uniform cannot be used because it is not stateless and
    #     # for some reason model.save_weights cannot handle that
    #     # tf.random.uniform_stateless introduces some weird subtle bug that I cannot figure out,
    #     # especially because the issue does not occur when it is not in pipeline.
    #     # numpy seems to introduce some new unexpected behavior.
    #     # Okay I think the reason numpy is trange is related to this: https://stackoverflow.com/q/69108284
    #     # The Dataset is executed once, which means that the same random number is used every time... ish
    #     def normalize(img, label):
    #         return tf.cast(img, tf.float32) / 255.0, label
    #
    #     def bernoullisample(img, label):
    #         return (
    #             tf.cast(tf.math.greater(img, tf.random.uniform(img.shape)), tf.float32),
    #             label,
    #         )
    #
    #     batch_size = 128
    #     data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    #     os.makedirs(data_dir, exist_ok=True)
    #
    #     (ds_train, ds_val, ds_test), ds_info = tfds.load(
    #         "mnist",
    #         split=["train", "test", "test"],
    #         shuffle_files=True,
    #         data_dir=data_dir,
    #         with_info=True,
    #         as_supervised=True,
    #     )
    #
    #     # ---- shuffling, infinite yield, preprocessing
    #     # https://stackoverflow.com/a/50453698
    #     # https://stackoverflow.com/a/49916221
    #     # https://www.tensorflow.org/guide/data#randomly_shuffling_input_data
    #     # https://www.tensorflow.org/datasets/overview
    #     # https://www.tensorflow.org/datasets/splits
    #     # manual dataset https://www.tensorflow.org/datasets/add_dataset
    #     # https://www.reddit.com/r/MachineLearning/comments/65me2d/d_proper_crop_for_celeba/
    #     # celeba hint: https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/train.py#L121
    #     # download celeba: https://github.com/AntixK/PyTorch-VAE
    #     ds_train = (
    #         ds_train.map(normalize, num_parallel_calls=4)
    #         .map(bernoullisample)
    #         .shuffle(len(ds_train))
    #         .repeat()
    #         .batch(batch_size)
    #         .prefetch(4)
    #     )
    #
    #     ds_val = (
    #         ds_val.map(normalize, num_parallel_calls=4)
    #         .repeat()
    #         .batch(10000)
    #         .prefetch(4)
    #     )
    #
    #     ds_test = ds_test.map(normalize).prefetch(4)
    #
    #     return iter(ds_train), iter(ds_val), ds_test


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib.pyplot as plt
    import numpy as np

    model = Model11()
    # ds = DataSets()
    # model.load("best")

    x, y = next(model.train_loader)
    xval, yval = next(model.val_loader)

    z, qzx, pxz = model(x, model.n_samples)

    model.save("test")

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

    # ---- debug data
    def normalize(img, label):
        prob = (
            tf.cast(img, tf.float32) / 255.0
        )  # crazy hard to debug error here if you don't cast the image first
        # img = tf.cast(tf.math.greater(prob, tf.random.stateless_uniform(prob.shape, seed=(4, 2))), tf.float32)
        img = tf.cast(tf.math.greater(prob, np.random.rand(*prob.shape)), tf.float32)
        # img = tf.cast(tf.math.greater(prob, 0.5), tf.float32)
        return img, label

    data_dir = None
    batch_size = 128
    data_dir = "/tmp/nsbi/data" if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test", "test"],
        shuffle_files=True,
        data_dir=data_dir,
        with_info=True,
        as_supervised=True,
    )

    # ---- this works
    Xtrain, ytrain = next(iter(ds_train.batch(len(ds_train))))
    Xval, yval = next(iter(ds_val.batch(len(ds_val))))
    Xtrain, ytrain = normalize(Xtrain, ytrain)
    Xval, yval = normalize(Xval, yval)

    # ---- this doesn't
    check, _ = next(iter(ds_train.map(normalize).batch(len(ds_train))))

    # ---- check if shuffling or repeat works
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
        ds_val.map(normalize, num_parallel_calls=4).repeat().batch(10000).prefetch(4)
    )

    loader = iter(ds_train)

    for x, y in loader:
        print(x[0, :, :, 0])
        break

    ds = np.linspace(1, 100, 100)

    ds_train = tf.data.Dataset.from_tensor_slices(ds).shuffle(100).repeat(2).batch(10)

    for x in ds_train:
        print(x)

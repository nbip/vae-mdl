"""
copied from:
https://github.com/rll/deepul/blob/master/deepul/hw3_helper.py
https://github.com/rll/deepul/blob/master/deepul/utils.py
https://github.com/rll/deepul/blob/master/homeworks/solutions/hw3_solutions.ipynb

also see
https://sites.google.com/view/berkeley-cs294-158-sp19/home
https://drive.google.com/file/d/1IrPBblLovAImcZdWnzJO07OxT7QD9X2m/view

"""

import os
import pickle
from collections import OrderedDict
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import make_grid
from tqdm import tqdm


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.deconvs(out)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std


class ConvVAE(nn.Module):
    def __init__(self, input_shape, latent_size):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = ConvEncoder(input_shape, latent_size)
        self.decoder = ConvDecoder(latent_size, input_shape)

    def loss(self, x):
        x = 2 * x - 1
        mu, log_std = self.encoder(x)
        z = torch.randn_like(mu) * log_std.exp() + mu
        x_recon = self.decoder(z)

        recon_loss = (
            F.mse_loss(x, x_recon, reduction="none").view(x.shape[0], -1).sum(1).mean()
        )
        kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        return OrderedDict(
            loss=recon_loss + kl_loss, recon_loss=recon_loss, kl_loss=kl_loss
        )

    def sample(self, n):
        with torch.no_grad():
            if torch.cuda.is_available():
                z = torch.randn(n, self.latent_size).cuda()
            else:
                z = torch.randn(n, self.latent_size)
            samples = torch.clamp(self.decoder(z), -1, 1)
        return samples.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5


def q2_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    """ YOUR CODE HERE """

    train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255.0).astype("float32")
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.0).astype("float32")

    if torch.cuda.is_available():
        model = ConvVAE((3, 32, 32), 16).cuda()
    else:
        model = ConvVAE((3, 32, 32), 16)

    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(
        model, train_loader, test_loader, dict(epochs=20, lr=1e-3), quiet=False
    )
    train_losses = np.stack(
        (train_losses["loss"], train_losses["recon_loss"], train_losses["kl_loss"]),
        axis=1,
    )
    test_losses = np.stack(
        (test_losses["loss"], test_losses["recon_loss"], test_losses["kl_loss"]), axis=1
    )
    samples = model.sample(100) * 255.0

    if torch.cuda.is_available():
        x = next(iter(test_loader))[:50].cuda()
    else:
        x = next(iter(test_loader))[:50]
    with torch.no_grad():
        x = 2 * x - 1
        z, _ = model.encoder(x)
        x_recon = torch.clamp(model.decoder(z), -1, 1)
    reconstructions = torch.stack((x, x_recon), dim=1).view(-1, 3, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu().numpy() * 255

    if torch.cuda.is_available():
        x = next(iter(test_loader))[:20].cuda()
    else:
        x = next(iter(test_loader))[:20]

    with torch.no_grad():
        x = 2 * x - 1
        z, _ = model.encoder(x)
        z1, z2 = z.chunk(2, dim=0)
        interps = [
            model.decoder(z1 * (1 - alpha) + z2 * alpha)
            for alpha in np.linspace(0, 1, 10)
        ]
        interps = torch.stack(interps, dim=1).view(-1, 3, 32, 32)
        interps = torch.clamp(interps, -1, 1) * 0.5 + 0.5
    interps = interps.permute(0, 2, 3, 1).cpu().numpy() * 255

    return train_losses, test_losses, samples, reconstructions, interps


def q2_save_results(part, dset_id, fn):
    """This is the main function that runs everything!"""
    assert part in ["a", "b"] and dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "svhn.pkl"))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, "cifar10.pkl"))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(
        train_data, test_data, dset_id
    )
    samples, reconstructions, interpolations = (
        samples.astype("float32"),
        reconstructions.astype("float32"),
        interpolations.astype("float32"),
    )
    print(
        f"Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, "
        f"KL Loss: {test_losses[-1, 2]:.4f}"
    )
    plot_vae_training_plot(
        train_losses,
        test_losses,
        f"Q2({part}) Dataset {dset_id} Train Plot",
        f"results/q2_{part}_dset{dset_id}_train_plot.png",
    )
    show_samples(
        samples,
        title=f"Q2({part}) Dataset {dset_id} Samples",
        fname=f"results/q2_{part}_dset{dset_id}_samples.png",
    )
    show_samples(
        reconstructions,
        title=f"Q2({part}) Dataset {dset_id} Reconstructions",
        fname=f"results/q2_{part}_dset{dset_id}_reconstructions.png",
    )
    show_samples(
        interpolations,
        title=f"Q2({part}) Dataset {dset_id} Interpolations",
        fname=f"results/q2_{part}_dset{dset_id}_interpolations.png",
    )


def show_samples(samples, fname=None, nrow=10, title="Samples"):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def get_data_dir(hw_number):
    return "data"


def load_pickled_data(fname, include_labels=False):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    train_data, test_data = data["train"], data["test"]
    if "mnist.pkl" in fname or "shapes.pkl" in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype("uint8")
        test_data = (test_data > 127.5).astype("uint8")
    if "celeb.pkl" in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data["train_labels"], data["test_labels"]
    return train_data, test_data


def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = (
        train_losses[:, 0],
        train_losses[:, 1],
        train_losses[:, 2],
    )
    elbo_test, recon_test, kl_test = (
        test_losses[:, 0],
        test_losses[:, 1],
        test_losses[:, 2],
    )
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label="-elbo_train")
    plt.plot(x_train, recon_train, label="recon_loss_train")
    plt.plot(x_train, kl_train, label="kl_loss_train")
    plt.plot(x_test, elbo_test, label="-elbo_test")
    plt.plot(x_test, recon_test, label="recon_loss_test")
    plt.plot(x_test, kl_test, label="kl_loss_test")

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    savefig(fname)


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    epochs, lr = train_args["epochs"], train_args["lr"]
    grad_clip = train_args.get("grad_clip", None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return train_losses, test_losses


def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        if torch.cuda.is_available():
            x = x.cuda()
        out = model.loss(x)
        optimizer.zero_grad()
        out["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f"Epoch {epoch}"
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f", {k} {avg_loss:.4f}"

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = "Test "
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f", {k} {total_losses[k]:.4f}"
        if not quiet:
            print(desc)
    return total_losses


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("cuda? ", torch.cuda.is_available())

    q2_save_results("a", 1, q2_a)

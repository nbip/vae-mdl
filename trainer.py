import tqdm

from models.model import Model


def train(model: Model, n_updates=int(1e6), eval_interval=1000):
    best = float("inf")
    for i in tqdm.tqdm(range(n_updates)):
        _, train_metrics = model.train_batch()
        if i % eval_interval == 0:
            val_loss, val_metrics = model.val_batch()
            model.save("latest")
            if val_loss < best:
                best = val_loss
                model.save("best")

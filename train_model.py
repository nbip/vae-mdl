from trainer import train
from models.model01 import Model01


if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python train_model.py > log.log &
    model = Model01()
    model.val_batch()
    train(model, n_updates=10_000, eval_interval=100)

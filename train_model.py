from models.model01 import Model01
from models.model02 import Model02
from models.model03 import Model03
from models.model04 import Model04
from models.model05 import Model05
from trainer import train

if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u train_model.py > train_model.log &
    model = Model05()
    model.val_batch()
    train(model, n_updates=1_000_000, eval_interval=1000)

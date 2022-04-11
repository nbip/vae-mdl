from models.model01 import Model01
from models.model02 import Model02
from models.model03 import Model03
from models.model04 import Model04
from models.model08 import Model08
from models.model09 import Model09
from models.model10 import Model10
from models.model11 import Model11
from models.model12 import Model12
from models.model14 import Model14
from trainer import train

if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u train_model.py > train_model.log &
    model = Model14()
    model.val_batch()
    train(model, n_updates=1_000_000, eval_interval=1000)

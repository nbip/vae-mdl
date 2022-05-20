from models.model20 import Model20
from models.model21 import Model21
from models.model23 import Model23
from models.model32 import Model32
from models.model35 import Model35
from models.model36 import Model36
from models.model37 import Model37
from models.model38 import Model38
from models.model40 import Model40
from models.model42 import Model42
from models.model43 import Model43
from models.model44 import Model44
from models.model47 import Model47
from models.model48 import Model48
from models.model49 import Model49
from trainer import train

if __name__ == "__main__":
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python -u train_model.py > train_model.log &
    model = Model49()
    model.val_batch()
    train(model, n_updates=1_000_000, eval_interval=1000)

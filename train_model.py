import logging

from numba import cuda

logging.basicConfig(level=logging.INFO)

import sys
sys.path.extend([".", ".."])
from nebula import Nebula


# model setup
TOKENIZER = "bpe"
nebula = Nebula(
    vocab_size = 50000,
    seq_len = 512,
    tokenizer = TOKENIZER,
)


TRAIN_SAMPLE = True
if TRAIN_SAMPLE:
    # print("#1")
    from torch.optim import AdamW
    from torch.nn import BCEWithLogitsLoss
    from nebula import ModelTrainer

    # print("#2")
    TIME_BUDGET = 5 # minutes
    device = "cuda" if cuda.is_available() else "cpu"
    model_trainer_config = {
        "device": device,
        "model": nebula.model,
        "loss_function": BCEWithLogitsLoss(),
        "optimizer_class": AdamW,
        "optimizer_config": {"lr": 2.5e-4, "weight_decay": 1e-2},
        "optim_scheduler": None, # supports multiple LR schedulers
        "optim_step_budget": None,
        "outputFolder": "out",
        "batchSize": 96,
        "verbosity_n_batches": 100,
        "clip_grad_norm": 1.0,
        "n_batches_grad_update": 1,
        "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
    }
    # print("#3")

    # 3. TRAIN
    model_trainer = ModelTrainer(**model_trainer_config)
    model_trainer.train(x, y)

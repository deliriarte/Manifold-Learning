import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar
from Datasets import dataset

import os
import random

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback, used for storing metrics after each validation epoch."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append({key: val.item() for (key, val) in trainer.callback_metrics.items()})
        

class LitProgressBar(ProgressBar):
    """Disable validation ProgressBar in pytorch-lightning,
    which leads to a bugged output inside a jupyter notebook for some reason"""

    def init_validation_tqdm(self):
        bar = tqdm(            
            disable=True,            
        )
        return bar
    
        
def cross_validate(features : 'torch.tensor',
                   batch_size : int = 10,
                   folds : int = 5,
                   random_state : int = 1):
    """
    Generate DataLoader of train/validation datasets for cross validation.
    
    The dataset specified by (`features`, `labels`) is split into a number of folds given by `folds`. 
    Then samples are batched by DataLoaders according to `batch_size`. 
    """

    kf = KFold(n_splits = folds, random_state = random_state, shuffle = True)

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
        x_train, x_val = features[train_idx], features[val_idx]

        train = dataset(x_train)
        val   = dataset(x_val)
        
        #Dataloaders
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
        val_loader   = torch.utils.data.DataLoader(val,   batch_size = batch_size, shuffle = True)

        yield (train_loader, val_loader)
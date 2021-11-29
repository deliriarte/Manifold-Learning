import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class dataset(Dataset):
    def __init__(self, elem):      
        mat_torch = torch.from_numpy(np.asanyarray(elem))
        self.fc =  torch.unsqueeze(mat_torch,1)

    def __getitem__(self, index):
        x = self.fc[index]
        return x
        
    def __len__(self):
        return len(self.fc)
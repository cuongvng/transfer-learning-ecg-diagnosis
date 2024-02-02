import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class PTBXLPartition(Dataset):
    CLASSES = ['NORM', 'MI', 'STTC', 'HYP', 'CD']
    
    def __init__(self, purpose, data_dir='./data/', training_size=None):
        assert purpose in ['train', 'val', 'test']
        
            
        x_dir = os.path.join(data_dir, f'x_{purpose}.csv')
        y_dir = os.path.join(data_dir, f'y_{purpose}.csv')
        
        self.x = pd.read_csv(x_dir)
        self.Y = pd.read_csv(y_dir)[['ecg_id'] + self.CLASSES]
        self.y = self.Y[self.CLASSES].values
        
        if purpose == 'train':
            self.Y = self.Y[:training_size]
        
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, index: int):
        ecg_id = self.Y['ecg_id'].iloc[index]
        # labels: One-hot values
        labels = torch.tensor(
            self.Y[self.CLASSES].iloc[index].values
        )
        
        # signals: 2-D array of size (1000, 12) representing 12-lead signals of length 1000
        signals = self.x[self.x.ecg_id == ecg_id].drop(columns=['ecg_id']).values
        
        signals = torch.tensor(signals)
        
        return (signals, labels)

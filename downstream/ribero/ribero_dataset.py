from torch.utils.data import Dataset
import joblib
import os

class RiberoDataset(Dataset):
    CLASSES = ['NORM', '1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

    def __init__(self, purpose, data_dir='./data'):
        assert purpose in ['train', 'test']
        x_dir = os.path.join(data_dir, f'X_{purpose}.joblib')
        y_dir = os.path.join(data_dir, f'y_{purpose}.joblib')

        self.X = joblib.load(x_dir)
        self.y = joblib.load(y_dir)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index: int):
        # labels: One-hot values
        labels = self.y[index]

        # signals: 2-D array of size (1000, 12) representing 12-lead signals of length 1000
        signals = self.X[index]
        
        return (signals, labels)

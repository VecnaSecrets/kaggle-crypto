import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class Pipeline_lstm:
    """
    Пайплайн для некого пандаса одной монеты уже с выброшенными лишними фичами.
    """
    def __init__(self, impute_strategy='mean'):
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = MinMaxScaler()

    def fit(self, X_raw: pd.DataFrame, y_raw: pd.Series):
        self.imputer.fit(X_raw)
        self.scaler.fit(X_raw)

    def transform(self, X_raw: pd.DataFrame, y_raw=None, lookback=10) -> TensorDataset:
        res = pd.DataFrame(self.imputer.transform(X_raw), columns=X_raw.columns)
        res = pd.DataFrame(self.scaler.transform(res), columns=res.columns)
        res = res.values
        if y_raw is not None:
            y = y_raw.values

        X_organized, Y_organized = [], []
        for i in range(0, res.shape[0]-lookback, 1):
            X_organized.append(res[i:i+lookback])
            if y_raw is not None:
                Y_organized.append(y[i+lookback])

        X_organized = torch.tensor(np.array(X_organized), dtype=torch.float32, device='cpu')
        if y_raw is not None:
            Y_organized = torch.tensor(np.array(Y_organized), dtype=torch.float32, device='cpu')

        if y_raw is not None:
            return TensorDataset(X_organized, Y_organized)
        else:
            return TensorDataset(X_organized)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args)
        return self.transform(*args, **kwargs)
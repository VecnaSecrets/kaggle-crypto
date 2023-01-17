import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def organize(*arrs: np.array, lookback=5):
    for ind, n in enumerate(arrs):
        if len(n.shape) == 1:
            print(f"expected 2D data but {ind} array is {len(n.shape)} dimentioned")
            raise Exception

    res = [np.zeros((
        n.shape[0],
        lookback,
        n.shape[1]
    )) for n in arrs]

    for ind, df in enumerate(res):
        for i in range(lookback-1):
            df[i] = (arrs[ind][i].reshape((1, -1)).T @ np.ones((1,lookback))).T

    for ind, df in enumerate(res):
        for i in range(lookback-1, df.shape[0]):
            df[i] = arrs[ind][i-lookback+1:i+1]

    return (np.array(n) for n in res)


class Pipeline_lstm:
    """
    Пайплайн для некого пандаса одной монеты уже с выброшенными лишними фичами.
    """
    def __init__(self, impute_strategy='mean', lookback=10):
        self.lookback=lookback
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = MinMaxScaler()

    def fit(self, X_raw: pd.DataFrame, y_raw: pd.Series = None):
        self.imputer.fit(X_raw)
        self.scaler.fit(X_raw)

    def transform(self, X_raw: pd.DataFrame, y_raw=None) -> TensorDataset:
        res = self.imputer.transform(X_raw)
        res = self.scaler.transform(res)
        # res = res.values
        if y_raw is not None:
            y = np.array(y_raw).reshape(-1,1)

        X_organized = organize(res, lookback=self.lookback).__next__()

        X_organized = torch.tensor(X_organized, dtype=torch.float32, device='cpu')
        if y_raw is not None:
            y = torch.tensor(y, dtype=torch.float32, device='cpu')

        if y_raw is not None:
            return TensorDataset(X_organized, y)
        else:
            return TensorDataset(X_organized)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args)
        return self.transform(*args, **kwargs)


if __name__ == '__main__':
    a = np.arange(0,15).reshape(5,3)
    b = np.arange(10,15).reshape(-1,1)
    print(organize(a, lookback=2).__next__())
# print(inspect.signature(Pipeline_lstm.fit).parameters.keys())
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import inspect

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
            print(df[i])
            df[i] = arrs[ind][i-lookback+1:i+1]

    return (np.array(n) for n in res)


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
            y = y_raw.values.reshape(-1,1)
        if y_raw is not None:
            X_organized, Y_organized = organize(res, y, lookback=lookback)
        else:
            X_organized = organize(res, lookback=lookback)


        # X_organized, Y_organized = [], []
        # for i in range(lookback):
        #     X_organized.append([res[i]]*lookback)
        #     if y_raw is not None:
        #         Y_organized.append(y[i])
        #
        # for i in range(lookback, res.shape[0], 1):
        #     X_organized.append(res[i-lookback:i])
        #     if y_raw is not None:
        #         Y_organized.append(y[i])

        X_organized = torch.tensor(X_organized, dtype=torch.float32, device='cpu')
        if y_raw is not None:
            Y_organized = torch.tensor(Y_organized, dtype=torch.float32, device='cpu')

        if y_raw is not None:
            return TensorDataset(X_organized, Y_organized)
        else:
            return TensorDataset(X_organized)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args)
        return self.transform(*args, **kwargs)


if __name__ == '__main__':
    a = np.arange(0,15).reshape(5,3)
    b = np.arange(10,15).reshape(-1,1)
    print(organize(a, b, lookback=2))
# print(inspect.signature(Pipeline_lstm.fit).parameters.keys())
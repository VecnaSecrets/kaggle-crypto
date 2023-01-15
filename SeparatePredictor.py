from train_tools import CalcValLoss, CalcR2_MSE_score, TrainModel
from utils import Pipeline_lstm
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

import datetime

class SeparatePredictor:
    def __init__(self, models: list,
                 data: pd.DataFrame,
                 lookback=10,
                 feat_devide="coin_id",
                 target = "fwd_ret_3d",
                 relevant_features=['feat_1', 'feat_9', 'feat_10'],
                 train_size = 0.8,
                 batch_size = 32,
                 **kwargs):
        self.num_classes = data[feat_devide].unique().shape[0]
        self.raw_data = data.drop(target, axis=1)
        self.class_labels = []
        self.y = data[target]
        self.devided_data = []
        self.y_devided = []
        self.lookback=lookback
        self.other_args = kwargs
        self.relevant_features = relevant_features
        self.feat_devide = feat_devide
        self.train_size=train_size
        self.batch_size = batch_size

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        if len(models) == 1:
            self.models = models*self.num_classes
        elif len(models) != self.num_classes:
            print("Error, num of models do not correspond to num of classes")
            raise Exception
        else:
            self.models = models

        self.set_up_pipeline()
        self.prepare()

    def set_up_pipeline(self):
        """
        here we will define our preprocess pipeline as well as training functions
        :return:
        """
        self.preprocess_p = [Pipeline_lstm()]*self.num_classes
        self.train_f = TrainModel
        self.validate_f = CalcR2_MSE_score
        pass

    def prepare(self):
        self.devided_data, self.y_devided = self.separate_data()
        self.train_test_split()
        self.fit_preprocess()

        self.train_ds = self.prepare_data(self.X_train, self.y_train)
        self.test_ds = self.prepare_data(self.X_test, self.y_test)

        self.train_loader = self.form_loaders(self.train_ds, self.batch_size)
        self.test_loader = self.form_loaders(self.test_ds, self.batch_size)


    def fit_preprocess(self):
        for ind, n in enumerate(self.X_train):
            self.preprocess_p[ind].fit(X_raw=n)

    def form_loaders(self, listed_ds, batch_size=32):
        res = []
        for ind, n in enumerate(listed_ds):
            res.append(DataLoader(listed_ds[ind], shuffle=False, batch_size=batch_size))
        return res

    def separate_data(self) -> list:
        res = []
        y = []
        self.class_labels = []
        for n in self.raw_data[self.feat_devide].unique():
            res.append(self.raw_data.loc[
                           self.raw_data[self.feat_devide] == n
                           ][self.relevant_features]
                       .values)
            self.class_labels.append(n)
            y.append(self.y[
                         self.raw_data[self.feat_devide] == n
                         ])
        return res, y

    def prepare_data(self, listed_data, listed_y=None):
        res = []
        for ind, n in enumerate(listed_data):
            res.append(self.preprocess_p[ind].transform(n, listed_y[ind]))
        return res

    def train_test_split(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        for ind, n in enumerate(self.devided_data):
            train_ind = int(self.devided_data[ind].shape[0]*self.train_size)
            self.X_train.append(self.devided_data[ind][:train_ind])
            self.X_test.append(self.devided_data[ind][train_ind:])
            self.y_train.append(self.y_devided[ind][:train_ind])
            self.y_test.append(self.y_devided[ind][train_ind:])
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, device='cpu', **kwargs):
        for ind, n in enumerate(self.train_loader):
            print(f"TRAINING f{self.class_labels[ind]}")
            self.train_class(self.models[ind],
                             self.train_loader[ind],
                             self.test_loader[ind],
                             device=device,
                             **kwargs)

            r2, mse = self.validate_f(self.models[ind], self.test_ds[ind], device)
            print(f"Total r2 score {r2}")
            print(f"Total mse score {mse}")
            self.save_model(self.models[ind], self.class_labels[ind])

    def save_model(self, model, label='other', folder='./models'):
        time = datetime.datetime.now().strftime("%d-%m-%y:%H:%M:%S")
        if f'{label}' not in os.listdir(folder):
            os.mkdir(f'{folder}/{label}/')

        MODEL_PATH = f'{folder}/{label}/{time}_lstm.pt'
        torch.save(model.state_dict(), MODEL_PATH)

    def train_class(self,
                    model,
                    train_loader,
                    test_loader,
                    epochs=2000,
                    learning_rate=1e-4,
                    loss_fn=None,
                    optimizer_cl=None,
                    optimizer_params={},
                    device='cpu',
                    writer=None,
                    other_train_params={}):
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer_cl is None:
            optimizer = Adam(model.parameters(), lr=learning_rate, **optimizer_params)
        else:
            optimizer = optimizer_cl(model.parameters(), **optimizer_params)

        _ = self.train_f(model, loss_fn, optimizer, train_loader, test_loader, epochs, device=device,writer=writer, **other_train_params)

    def predict(self, X):
        pass

from train_tools import LSTMRegressor
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_regressor = LSTMRegressor(3,
                                   32,
                                   2,
                                   device=DEVICE).to(device=DEVICE)
    df = df = pd.read_csv('train.csv')

    predictor = SeparatePredictor(models = [lstm_regressor], data=df)
    predictor.train(device=DEVICE, epochs=10)

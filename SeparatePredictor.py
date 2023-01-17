from train_tools import CalcValLoss, CalcR2_MSE_score, TrainModel
from utils import Pipeline_lstm
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np

import datetime

class SeparatePredictor:
    def __init__(self, models: list,
                 data: pd.DataFrame,
                 params: dict = {},
                 lookback=10,
                 feat_devide="coin_id",
                 target = "fwd_ret_3d",
                 relevant_features=['feat_1', 'feat_9', 'feat_10'],
                 train_size = 0.8,
                 batch_size = 32,
                 device='cpu',
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
        self.device='cpu'

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        if len(models) == 1:
            models_arr = []
            for i in range(self.num_classes):
                models_arr.append(models[0](**params))
            self.models = models_arr
        elif len(models) != self.num_classes:
            print("Error, num of models do not correspond to num of classes")
            raise Exception
        else:
            self.models = models



    def set_up_pipeline(self,
                        pipeline_f,
                        pipe_params={},
                        train_f=None,
                        validate_f=None):
        """
        here we will define our preprocess pipeline as well as training functions
        :return:
        """
        self.preprocess_p = dict(zip(self.class_labels, [pipeline_f(**pipe_params) for _ in range(self.num_classes)]))
        self.train_f = train_f
        self.validate_f = validate_f
        pass

    def prepare(self):
        self.devided_data, self.y_devided, self.class_labels = self.separate_data(self.raw_data, self.y)
        self.models = dict(zip(self.class_labels, self.models))
        self.set_up_pipeline(pipeline_f=Pipeline_lstm,
                             train_f=TrainModel,
                             validate_f=CalcValLoss())
        self.train_test_split()
        self.fit_preprocess()

        self.train_ds = self.prepare_data(self.X_train, self.y_train)
        self.test_ds = self.prepare_data(self.X_test, self.y_test)

        self.train_loader = self.form_loaders(self.train_ds, self.batch_size)
        self.test_loader = self.form_loaders(self.test_ds, self.batch_size)


    def fit_preprocess(self):
        for n in self.class_labels:
            self.preprocess_p[n].fit(X_raw=self.X_train[n])

    def form_loaders(self, listed_ds, batch_size=32):
        res = {}
        for n in self.class_labels:
            res[n] = DataLoader(listed_ds[n], shuffle=False, batch_size=batch_size)
        return res

    def separate_data(self, raw_data, y_raw=None) -> list:
        res = {}
        y = {}
        class_labels = raw_data[self.feat_devide].unique().tolist()
        for n in class_labels:
            res[n] = raw_data.loc[
                raw_data[self.feat_devide] == n
                ][self.relevant_features] \
                .values
            if y_raw is not None:
                y[n] = y_raw[raw_data[self.feat_devide] == n]
        if y_raw is not None:
            return res, y, class_labels
        return res, class_labels

    def prepare_data(self, listed_data, listed_y=None, transform_params={}):
        res = {}
        for n in self.class_labels:
            if listed_y is not None:
                res[n] = self.preprocess_p[n].transform(listed_data[n], listed_y[n], **transform_params)
            else:
                res[n] = self.preprocess_p[n].transform(listed_data[n], **transform_params)
        return res

    def train_test_split(self):
        self.X_train = {}
        self.X_test = {}
        self.y_train = {}
        self.y_test = {}

        for cls in self.class_labels:
            train_ind = int(self.devided_data[cls].shape[0]*self.train_size)
            self.X_train[cls] = self.devided_data[cls][:train_ind]
            self.X_test[cls] = self.devided_data[cls][train_ind:]
            self.y_train[cls] = self.y_devided[cls][:train_ind]
            self.y_test[cls] = self.y_devided[cls][train_ind:]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, device='cpu', **kwargs):
        for cls in self.class_labels:
            print(f"TRAINING f{cls}")
            self.train_class(self.models[cls],
                             self.train_loader[cls],
                             self.test_loader[cls],
                             device=device,
                             **kwargs)

            r2, mse = self.validate_f(self.models[cls], self.test_ds[cls], device)
            print(f"Total r2 score {r2}")
            print(f"Total mse score {mse}")
            self.save_model(self.models[cls], cls)

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

    def predict(self, X:pd.DataFrame, device='cpu', transform_params={}):
        res = pd.DataFrame(index=X.index, data=np.zeros(X.shape[0]))

        X_separated, classes = self.separate_data(X)
        X_ds = self.prepare_data(X_separated)
        out = {}
        with torch.no_grad():
            for cls in classes:
                out[cls] = self.models[cls].to(device=device) \
                    (X_ds[cls].tensors[0].to(device=device)) \
                    .to(device='cpu').detach().numpy()
                res.loc[X[self.feat_devide] == cls] = out[cls]

        return res

class Conv1dPredictor(SeparatePredictor):
    def prepare(self):
        self.devided_data, self.y_devided, self.class_labels = self.separate_data(self.raw_data, self.y)
        self.models = dict(zip(self.class_labels, self.models))
        self.set_up_pipeline(
            pipeline_f=Pipeline_lstm,
            pipe_params={'lookback':self.lookback},
            train_f=TrainModel_NoLoader
        )
        self.train_test_split()
        self.fit_preprocess()

        self.train_ds = self.prepare_data(self.X_train, self.y_train)
        self.test_ds = self.prepare_data(self.X_test, self.y_test)

    def train(self, device='cpu', epoch_shedule=None, epochs=100, **kwargs):
        for cls in self.class_labels:
            print(f"TRAINING f{cls}")
            writer = SummaryWriter('./runs/1dConv_multi/{}'.format(cls))
            if epoch_shedule is not None:
                epochs = epoch_shedule[cls]
            self.train_class(self.models[cls].to(device=device),
                             self.train_ds[cls],
                             self.test_ds[cls],
                             device=device,
                             writer=writer,
                             epochs=epochs,
                             **kwargs)


from train_tools import LSTMRegressor
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('train.csv')
    df_t = pd.read_csv('test.csv')
    params = {
        'input_size': 3,
        'hidden_dim':32,
        'n_layers': 2,
        'device':DEVICE
    }
    predictor = SeparatePredictor(models = [LSTMRegressor],
                                  params=params,
                                  data=df,
                                  device=DEVICE)
    predictor.prepare()
    predictor.train(device=DEVICE, epochs=5)
    preds = predictor.predict(df_t, device=DEVICE)
    print(df_t.shape)
    print(preds.shape)

from tqdm import tqdm
import torch
import torch as t
from sklearn.metrics import r2_score, mean_squared_error

def CalcValLoss(model, loss_fn, val_loader, device='cpu', return_preds=False):
    with torch.no_grad():
        losses = []
        preds_all = torch.tensor([], dtype=torch.float32, device='cpu')
        for X, Y in val_loader:
            preds = model(X.to(device=device)).to(device='cpu')
            loss = loss_fn(preds.ravel(), Y)
            losses.append(loss.item())
            if (return_preds):
                preds_all = torch.concat([preds_all, preds], dim=0)
        avg_loss = torch.tensor(losses).mean()
        if (return_preds):
            return avg_loss, preds_all
        return avg_loss

def CalcR2_MSE_score(model, val_ds, device='cpu'):
    with torch.no_grad():
        preds_all = model.forward(val_ds.tensors[0].to(device=device)).to(device='cpu')
        preds_all = preds_all.detach().numpy()
        y_true = val_ds.tensors[1].detach().numpy()
    return \
        r2_score(y_true, preds_all), \
        mean_squared_error(y_true, preds_all)


def TrainModel(model,
               loss_fn,
               optimizer,
               train_loader,
               val_loader,
               epochs=10,
               display_on_epoch = 100,
               device='cpu',
               writer=None):
    pbar = tqdm(range(1, epochs+1))
    for i in pbar:
        losses = []
        for X, Y in train_loader:
            X = X.to(device=device)
            Y_preds = model(X)
            Y_preds = Y_preds.to(device='cpu')

            loss = loss_fn(Y_preds.ravel(), Y).to(device='cpu')
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % display_on_epoch == 0:
            avg_train_loss = torch.tensor(losses).mean()
            avg_val_loss = CalcValLoss(model, loss_fn, val_loader, device=device)
            if writer is not None:
                writer.add_scalar("train/loss", avg_train_loss, i)
                writer.add_scalar("val/loss", avg_val_loss, i)
            pbar.set_description("Train Loss: {:.2f}; Val Loss: {:.2f}".format(avg_train_loss, avg_val_loss))
    return losses

from torch import nn
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_dim=32, n_layers=2, device='cpu'):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True).to(device=self.device)
        self.linear = nn.Linear(self.hidden_dim, 1).to(device=self.device)

    def forward(self, X_batch):
        hidden, carry = torch.randn(self.n_layers, len(X_batch), self.hidden_dim, device=self.device), torch.randn(self.n_layers, len(X_batch), self.hidden_dim, device=self.device)
        output, (hidden, carry) = self.lstm(X_batch, (hidden, carry))
        return self.linear(output[:,-1])


def TrainModel_NoLoader(model,
                        loss_fn,
                        optimizer,
                        train_ds,
                        val_ds,
                        epochs=10,
                        display_on_epoch = 100,
                        device='cpu',
                        writer=None):
    pbar = tqdm(range(1, epochs+1))
    X_val = val_ds.tensors[0]
    for i in pbar:
        losses = []
        X = train_ds.tensors[0].to(device=device)
        Y_preds = model(X)
        Y_preds = Y_preds.to(device='cpu')

        loss = loss_fn(Y_preds, train_ds.tensors[1]).to(device='cpu')
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % display_on_epoch == 0:
            with t.no_grad():
                avg_train_loss = t.tensor(losses).mean()

                avg_val_loss = loss_fn(
                    model(X_val.to(device=device)).to(device='cpu'),
                    val_ds.tensors[1]
                )
            if writer is not None:
                writer.add_scalar("train/loss", avg_train_loss, i)
                writer.add_scalar("val/loss", avg_val_loss, i)
            pbar.set_description("Train Loss: {:.2f}; Val Loss: {:.2f}".format(avg_train_loss, avg_val_loss))
    return losses

def pearson_c(y_true: t.tensor, y_preds: t.tensor):
    x = t.concat([
        y_true.T,
        y_preds.T
    ], dim=0)
    corr = t.scalar_tensor(1) - t.corrcoef(x)[0][1]
    return corr
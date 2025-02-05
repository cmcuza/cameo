import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time


def create_dataset(ts, lookback):
    X, y = [], []
    for i in range(len(ts) - 2*lookback):
        feature = ts[i:i+lookback, ...].tolist()
        target = ts[i + lookback:i+2*lookback, ...].tolist()
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_layer = 64
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_layer, num_layers=1, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_layer*2*24, hidden_layer*24)
        self.leaky = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_layer*24, 24)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the model to the appropriate device
        print('Device', self.device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.leaky(x)
        x = self.linear2(x)
        return x

    def fit_val(self, ts, lookback):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        x_train, y_train = create_dataset(ts, lookback)
        split = 0.9
        x_val = x_train[int(x_train.shape[0] * split):, :, np.newaxis]
        y_val = y_train[int(y_train.shape[0] * split):, :, np.newaxis]
        x_train = x_train[:int(x_train.shape[0] * split), :, np.newaxis]
        y_train = y_train[:int(y_train.shape[0] * split), :, np.newaxis]
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)

        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)

        train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=32)
        val_loader = data.DataLoader(data.TensorDataset(x_val, y_val), shuffle=True, batch_size=32)

        best_val_rmse = np.inf
        n_epochs = 100
        for epoch in range(n_epochs):
            self.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            if epoch % 5 != 0:
                continue

            self.eval()

            with torch.no_grad():
                val_rmse = 0.
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.forward(x_batch)
                    val_rmse += torch.sqrt(loss_fn(y_pred, y_batch)).item()

                val_rmse /= len(val_loader)

                if val_rmse > best_val_rmse:
                    print('Early stop, previous val was', best_val_rmse, 'now is', val_rmse)
                    break

                best_val_rmse = val_rmse

                print("Epoch %d: eval RMSE %.4f" % (epoch, best_val_rmse))

    def fit(self, ts, lookback):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        x_train, y_train = create_dataset(ts, lookback)
        x_train = x_train[:, :, np.newaxis]
        y_train = y_train[:, :, np.newaxis]
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)

        print(x_train.shape, y_train.shape)

        train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=32)

        n_epochs = 20
        for epoch in range(n_epochs):
            self.train()
            epoch_mse = 0
            start_time = time.time()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, torch.squeeze(y_batch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_mse += loss.item()

            print('EPOCH:', epoch, 'Mean MSE', epoch_mse/len(train_loader), 'Time:', time.time() - start_time, flush=True)

    def predict_future(self, ts, lookback):
        test_ts = ts[-lookback:, np.newaxis]
        self.eval()
        with torch.no_grad():
            x_input = torch.tensor([test_ts.tolist()]).to(self.device)
            y_pred = self.forward(x_input).detach().cpu().numpy()

        return y_pred




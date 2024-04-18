from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

import plots
import torch
import copy
import dill
import os

# Chose defalut device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def save(
         scaler,
         model,
         df_parity,
         df_loss,
         X_train,
         y_train,
         X_val=None,
         y_val=None,
         X_test=None,
         y_test=None,
         save_dir='./outputs',
         ):

    os.makedirs(save_dir, exist_ok=True)

    plots.generate(df_parity, df_loss, save_dir)

    torch.save(
               model,
               os.path.join(save_dir, 'model.pth')
               )

    if scaler is not None:
        dill.dump(scaler, open(os.path.join(save_dir, 'scaler.pkl'), 'wb'))

    df_parity.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    df_loss.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)

    for i in range(len(X_train)):

        X = X_train[i].cpu().detach()
        y = y_train[i]
        np.savetxt(os.path.join(
                                save_dir,
                                f'X_train_{i}.csv',
                                ), X, delimiter=',')
        np.savetxt(os.path.join(
                                save_dir,
                                f'y_train_{i}.csv',
                                ), y, delimiter=',')

        if X_val is not None:
            np.savetxt(
                       os.path.join(save_dir, f'X_validation_{i}.csv'),
                       X_val[i].cpu().detach(),
                       delimiter=',',
                       )

        if y_val is not None:
            np.savetxt(
                       os.path.join(save_dir, f'y_validation_{i}.csv'),
                       y_val[i],
                       delimiter=',',
                       )

        if X_test is not None:
            np.savetxt(
                       os.path.join(save_dir, f'X_test_{i}.csv'),
                       X_test[i].cpu().detach(),
                       delimiter=',',
                       )

        if y_test is not None:
            np.savetxt(
                       os.path.join(save_dir, f'y_test_{i}.csv'),
                       y_test[i],
                       delimiter=',',
                       )


def to_tensor(x):
    y = torch.FloatTensor(x).to(device)

    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    return y


def loader(X, y, batch_size=32, shuffle=True):

    data = TensorDataset(X, y)
    data = DataLoader(
                      data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      )

    return data


def transform(Xin, yin, scalers):

    Xout = []
    for i in range(len(Xin)):

        if scalers[i] is not None:
            X = scalers[i].transform(Xin[i])
        else:
            X = Xin[i]

        Xout.append(to_tensor(X))

    yout = list(map(to_tensor, yin))

    return Xout, yout


def pred(model, X, y, indxs):

    df = []
    with torch.no_grad():
        for indx in indxs:

            d = pd.DataFrame()
            d['y'] = y[indx].cpu().detach().view(-1)
            d['p'] = model(X[indx], indx).cpu().detach().view(-1)
            d['data'] = indx
            df.append(d)

    df = pd.concat(df)

    return df


def train(
          model,
          optimizer,
          losses,
          X_train,
          y_train,
          X_val=None,
          y_val=None,
          X_test=None,
          y_test=None,
          n_epochs=1000,
          batch_size=32,
          lr=1e-4,
          scalers=None,
          save_dir='outputs',
          print_n=100,
          ):

    # Conditions in case validation and/or test sets are supplied
    valcond = all([X_val is not None, y_val is not None])  # Val set
    testcond = all([X_test is not None, y_test is not None])  # Test set

    # Copy objects
    model = copy.deepcopy(model).to(device)
    scalers = copy.deepcopy(scalers)
    losses = copy.deepcopy(losses)

    optimizer = optimizer(model.parameters(), lr=lr)

    n_datasets = len(X_train)
    assert n_datasets == len(y_train)

    # Fit scaler
    if scalers is None:
        scalers = [None]*n_datasets
    else:
        for i, j in zip(scalers, X_train):
            if i is not None:
                i.fit(j)

    # Scale data
    X_train, y_train = transform(X_train, y_train, scalers)

    if valcond:
        X_val, y_val = transform(X_val, y_val, scalers)

    if testcond:
        X_test, y_test = transform(X_test, y_test, scalers)

    data = [loader(X, y, batch_size) for X, y in zip(X_train, y_train)]
    data = CombinedLoader(data, 'max_size')

    indxs = range(n_datasets)
    df_loss = []
    for epoch in range(1, n_epochs+1):

        model.train()

        for batch, _, _ in data:

            loss = 0.0
            for indx in indxs:

                if batch[indx] is None:
                    continue

                X = batch[indx][0]
                y = batch[indx][1]

                p = model(X, indx)
                loss += losses[indx](p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()

            for indx in indxs:
                y = y_train[indx]
                p = model(X_train[indx], indx)
                loss = losses[indx](p, y).item()

                d = (epoch, loss, indx, 'train')
                df_loss.append(d)

                if valcond:

                    y = y_val[indx]
                    p = model(X_val[indx], indx)
                    loss = losses[indx](p, y).item()

                    d = (epoch, loss, indx, 'validation')
                    df_loss.append(d)

        if epoch % print_n == 0:
            p = f'Epoch {epoch}/{n_epochs}: '
            if valcond:
                print(p+f'Validation loss {loss:.2f}')
            else:
                print(p+f'Train loss {loss:.2f}')

    # Loss curve
    columns = ['epoch', 'loss', 'data', 'set']
    df_loss = pd.DataFrame(df_loss, columns=columns)

    # Train parity
    df_parity = pred(model, X_train, y_train, indxs)
    df_parity['set'] = 'train'

    # Validation parity
    if valcond:

        df = pred(model, X_val, y_val, indxs)
        df['set'] = 'validation'
        df_parity = pd.concat([df_parity, df])

    # Test parity
    if testcond:

        df = pred(model, X_test, y_test, indxs)
        df['set'] = 'test'
        df_parity = pd.concat([df_parity, df])

    save(
         scalers,
         model,
         df_parity,
         df_loss,
         X_train,
         y_train,
         X_val,
         y_val,
         X_test,
         y_test,
         save_dir,
         )

    out = {
           'model': model,
           'scalers': scalers,
           'df_loss': df_loss,
           'df_parity': df_parity,
           }

    return out

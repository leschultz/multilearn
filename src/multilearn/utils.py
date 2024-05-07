from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset
from multilearn import plots
from joblib import dump

import pandas as pd
import numpy as np

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
         model,
         df_parity,
         df_loss,
         data,
         save_dir='./outputs',
         ):

    '''
    Save results of run.

    Args:
        model (object): The trained tensorflow model.
        df_parity (pd.DataFrame): The parity plot data.
        df_loss (pd.DataFrame): The learning curve data.
        data (dict): The data splits.
        save_dir (str): The location to save outputs.
    '''

    os.makedirs(save_dir, exist_ok=True)

    plots.generate(df_parity, df_loss, save_dir)

    torch.save(
               model,
               os.path.join(save_dir, 'model.pth')
               )

    df_parity.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    df_loss.to_csv(os.path.join(save_dir, 'loss_vs_epochs.csv'), index=False)

    for key, value in data.items():

        new_dir = os.path.join(save_dir, key)
        for k, v in value.items():
            if k == 'scaler':
                dump(v, os.path.join(new_dir, 'scaler.joblib'))

            elif k == 'loss':
                dill.dump(
                          v,
                          open(os.path.join(new_dir, 'loss.pkl'), 'wb'),
                          )

            elif ('X_' in k) or ('y_' in k):

                v = v.cpu()
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                else:
                    v = v.detach()

                np.savetxt(os.path.join(
                                        new_dir,
                                        f'{k}.csv',
                                        ), v, delimiter=',')


def to_tensor(x):
    '''
    Convert variable to tensor.

    Args:
        x (np.ndarray): The variable to convert.

    Returns:
        torch.FloatTensor: The converted variable.
    '''

    y = torch.FloatTensor(x).to(device)

    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    return y


def loader(X, y, batch_size=32, shuffle=True):
    '''
    A wrapper to load data for pytorch.

    Args:
        X (torch.FloatTensor): The features.
        y (torch.FloatTensor): The target values.
        batch_size (int): The size of the batch for gradient descent.
        shuffle (bool): Whether to shuffle data.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    '''

    data = TensorDataset(X, y)
    data = DataLoader(
                      data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      )

    return data


def pred(model, data):
    '''
    Function to generate parity plot data predictions.

    Args:
        model (object): The trained model.
        data (dict): The data splits.

    Returns:
        pd.DataFrame: Parity plot data.
    '''

    df = []
    with torch.no_grad():
        for key, value in data.items():

            for k, v in value.items():

                if 'X_' in k:
                    split = k.split('_')[1]
                    X = value[k]
                    y = value['y_'+split]
                    d = pd.DataFrame()
                    d['y'] = y.cpu().detach().view(-1)
                    d['p'] = model(X, key).cpu().detach().view(-1)
                    d['data'] = key
                    d['split'] = split
                    df.append(d)

    df = pd.concat(df)

    return df


def train(
          model,
          optimizer,
          data,
          n_epochs=1000,
          batch_size=32,
          lr=1e-4,
          save_dir='outputs',
          patience=np.inf,
          print_n=100,
          ):
    '''
    The training workflow for models.

    Args:
        model (object): The model to train/assess.
        optimizer (object): The torch optimizer
        data (dict): The data with splits.
        n_epochs (int): The number of epochs to train.
        batch_size (int): The size of the batch for gradient descent.
        lr (float): The learning rate.
        save_dir (str): The location to save outputs.
        patience (int): Stop training if no improvement after n epochs.
        print_n (int): The interval to print loss.

    Returns:
        dict: The trained model and plot data.
    '''

    # Copy objects
    model = copy.deepcopy(model).to(device)
    data = copy.deepcopy(data)

    optimizer = optimizer(model.parameters(), lr=lr)

    # Fit scalers
    for key, value in data.items():
        for k, v in value.items():
            if k == 'scaler':
                value['scaler'].fit(value['X_train'])
                break

    # Apply transforms when needed
    data_train = {}
    for key, value in data.items():
        for k, v in value.items():
            if ('X_' in k) and ('scaler' in value.keys()):
                value[k] = value['scaler'].transform(value[k])

            if all([k != 'scaler', k != 'loss', k != 'weight']):
                value[k] = to_tensor(value[k])

        data_train[key] = loader(
                                 value['X_train'],
                                 value['y_train'],
                                 batch_size,
                                 )

    data_train = CombinedLoader(data_train, 'max_size')

    df_loss = []
    no_improv = 0
    best_loss = float('inf')
    for epoch in range(1, n_epochs+1):

        model.train()

        for batch, _, _ in data_train:

            loss = 0.0
            for indx in data.keys():

                if batch[indx] is None:
                    continue

                X = batch[indx][0]
                y = batch[indx][1]

                p = model(X, indx)
                i = data[indx]['loss'](p, y)

                if 'weight' in data[indx].keys():
                    i *= data[indx]['weight']

                loss += i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()

            all_loss = 0.0
            for indx in data.keys():
                y = data[indx]['y_train']
                p = model(data[indx]['X_train'], indx)
                loss = data[indx]['loss'](p, y).item()

                split = 'train'
                d = (epoch, loss, indx, split)
                df_loss.append(d)

                if 'y_val' in data[indx].keys():

                    y = data[indx]['y_val']
                    p = model(data[indx]['X_val'], indx)
                    loss = data[indx]['loss'](p, y).item()

                    split = 'val'
                    d = (epoch, loss, indx, split)
                    df_loss.append(d)

                    all_loss += loss

                else:
                    all_loss += loss

        # Early stopping
        if all_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = all_loss
            no_improv = 0

        else:
            no_improv = 1

        if no_improv >= patience:
            break

        if epoch % print_n == 0:
            print(f'Epoch {epoch}/{n_epochs}: {split} loss {loss:.2f}')

    # Loss curve
    columns = ['epoch', 'loss', 'data', 'split']
    df_loss = pd.DataFrame(df_loss, columns=columns)

    # Train parity
    df_parity = pred(model, data)

    save(
         model,
         df_parity,
         df_loss,
         data,
         save_dir,
         )

    out = {
           'model': best_model,
           'df_parity': df_parity,
           'df_loss': df_loss,
           'data': data,
           }

    return out

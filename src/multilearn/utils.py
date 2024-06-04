from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from multilearn import plots, models
from pathlib import Path

import pandas as pd

import torch
import copy
import dill
import os


def load(locations, names=None, targets=None, drops=None):
    '''
    Load data included with the package.

    Args:
        locations (list): The loctions of data to load.
        names (list): The names for each dataset.
        targets (list): The name of the target variable for each of names.
        drops (list): A list of list for columns to drop.

    Returns:
        Dict: A dictionary with features, targets, and names.
    '''

    namescond = names is None
    dropscond = drops is not None

    data = {}
    for count in range(len(locations)):

        if namescond:
            name = count
        else:
            name = names[count]

        if dropscond:
            drop = drops[count]

        else:
            drop = None

        target = targets[count]

        df = locations[count]
        df = pd.read_csv(df)

        if drop is None:
            X = df.drop(target, axis=1).values
        else:
            X = df.drop([target]+drop, axis=1).values

        y = df[target].values

        data[name] = {}
        data[name]['X'] = X
        data[name]['y'] = y

    return data


def find(where, match):
    paths = list(map(str, Path(where).rglob(match)))
    return paths


def find_load(*args):

    paths = find(*args)
    df = []
    for path in paths:
        df.append(pd.read_csv(path))

    df = pd.concat(df)

    return df


def save(
         df_parity,
         df_loss,
         model=None,
         save_dir='./outputs',
         ):

    '''
    Save results of run.

    Args:
        model (object): The trained tensorflow model.
        df_parity (pd.DataFrame): The parity plot data.
        df_loss (pd.DataFrame): The learning curve data.
        save_dir (str): The location to save outputs.
    '''

    os.makedirs(save_dir, exist_ok=True)

    plots.generate(df_parity, df_loss, save_dir)

    df_parity.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    df_loss.to_csv(os.path.join(save_dir, 'loss_vs_epochs.csv'), index=False)

    if model is not None:
        dill.dump(model, open(os.path.join(save_dir, 'model.dill'), 'wb'))


def to_tensor(x, device):
    '''
    Convert variable to tensor.

    Args:
        x (np.ndarray): The variable to convert.
        device (str): The device.

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
                    y_pred = model.predict(X, key)
                    index = value['indx_'+split]

                    d = pd.DataFrame()
                    d['y'] = y
                    d['y_pred'] = y_pred
                    d['data'] = key
                    d['split'] = split
                    d['index'] = index
                    df.append(d)

    df = pd.concat(df)

    return df


def cv(
       data,
       model,
       optimizer,
       splitter,
       *args,
       **kwargs,
       ):

    data = copy.deepcopy(data)
    model = copy.deepcopy(model)

    save_dir = kwargs['save_dir']
    del kwargs['save_dir']

    val_size = kwargs['val_size']

    # Generate splits (need to have the same number of folds)
    splits = {
              'name': [],
              'fold': [],
              'train': [],
              'validation': [],
              'test': [],
              }

    for key, item in data.items():
        iterator = enumerate(splitter.split(item['X']), start=1)
        for fold, (train_indx, test_indx) in iterator:

            train_indx, val_indx = train_test_split(
                                                    train_indx,
                                                    test_size=val_size,
                                                    )

            splits['name'].append(key)
            splits['fold'].append(fold)
            splits['train'].append(train_indx)
            splits['validation'].append(val_indx)
            splits['test'].append(test_indx)

    splits = pd.DataFrame(splits)
    total = splits.fold.max()

    # Group splits by folds and then do CV
    for group, values in splits.groupby('fold'):

        # Copy so that weight do not leak from previous fold
        submodel = copy.deepcopy(model)

        # Spilt data into train and test sets
        subdata = copy.deepcopy(data.copy())
        for key, item in data.items():

            indx_train = values.loc[values['name'] == key, 'train']
            indx_val = values.loc[values['name'] == key, 'validation']
            indx_test = values.loc[values['name'] == key, 'test']

            indx_train = indx_train.values[0]
            indx_val = indx_val.values[0]
            indx_test = indx_test.values[0]

            # Train, validation, and test indexes
            subdata[key]['indx_train'] = indx_train
            subdata[key]['indx_val'] = indx_val
            subdata[key]['indx_test'] = indx_test

            # Get training set
            subdata[key]['X_train'] = data[key]['X'][indx_train, :]
            subdata[key]['y_train'] = data[key]['y'][indx_train]

            # Get validation set
            subdata[key]['X_val'] = data[key]['X'][indx_val, :]
            subdata[key]['y_val'] = data[key]['y'][indx_val]

            # Get test set
            subdata[key]['X_test'] = data[key]['X'][indx_test, :]
            subdata[key]['y_test'] = data[key]['y'][indx_test]

        status = f'Fold {group}/{total}'
        n_letters = len(status)

        print('+'*n_letters)
        print(f'Fold {group}/{total}')
        print('-'*n_letters)

        # Create train and create predictions per fold
        submodel = models.model_wrapper(model)

        # Fit and get learning curve data
        df_loss = submodel.fit(subdata, optimizer, **kwargs)
        df_loss['fold'] = group

        # Parity plot data
        df_parity = pred(submodel, subdata)
        df_parity['fold'] = group

        save(
             df_parity,
             df_loss,
             save_dir=os.path.join(save_dir, f'folds/fold_{group}'),
             )

    # Gather all splits and plot
    save(
         find_load(save_dir, 'predictions.csv'),
         find_load(save_dir, 'loss_vs_epochs.csv'),
         save_dir=os.path.join(save_dir, 'folds/together'),
         )


def full_fit(
             data,
             model,
             optimizer,
             *args,
             **kwargs,
             ):

    data = copy.deepcopy(data)
    model = copy.deepcopy(model)

    save_dir = kwargs['save_dir']
    del kwargs['save_dir']

    print('++++++++')
    print('Full Fit')
    print('--------')

    model = models.model_wrapper(model)

    # Make all data training
    for key, value in data.items():

        value['X_train'] = value['X']
        value['y_train'] = value['y']
        value['indx_train'] = list(range(value['y'].shape[0]))

        del value['X']
        del value['y']

    df_loss = model.fit(data, optimizer, **kwargs)
    df_parity = pred(model, data)

    save(
         df_parity,
         df_loss,
         model,
         save_dir=os.path.join(save_dir, 'full_fit'),
         )

    return model

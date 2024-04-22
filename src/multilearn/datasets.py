from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import pkg_resources
import os

data_path = pkg_resources.resource_filename('multilearn', 'data')


def splitter(X, y, names=None, train_size=1.0, val_size=0.0, test_size=0.0):
    '''
    Split list of data into train, validation, and test splits.

    Args:
        X (list): A list of features.
        y (list): A list of target values.
        names (list): A list of names for each dataset.
        train_size (float): The fraction of training data.
        val_size (float): The fraction of validation data.
        test_size (float): The fraction of test data.

    Returns:
        dict: A dictionary of data splits.
    '''

    n = len(X)
    if names is None:
        assert n == len(y)
    else:
        assert n == len(y) == len(names)

    data = {}
    for i in range(n):
        d = split(X[i], y[i], train_size, val_size, test_size)

        if names is None:
            data[i] = d
        else:
            data[names[i]] = d

    return data


def split(X, y, train_size=1.0, val_size=0.0, test_size=0.0):
    '''
    Split data into train, validation, and test splits.

    Args:
        X (np.ndarray): A list of features.
        y (np.ndarray): A list of target values.
        train_size (float): The fraction of training data.
        val_size (float): The fraction of validation data.
        test_size (float): The fraction of test data.

    Returns:
        dict: A dictionary of data splits.
    '''

    # Make sure data splits sum to 1
    assert train_size+val_size+test_size == 1.0, (
        'Split fractions must sum to 1'
    )

    if train_size+val_size < 1.0:
        test_size = 1.0-train_size-val_size

    elif train_size+test_size < 1.0:
        val_size = 1.0-train_size-test_size

    elif val_size+test_size < 1.0:
        train_size = 1.0-val_size+test_size

    # Now split data as needed
    data = {}
    if train_size == 1.0:
        data['X_train'] = X
        data['y_train'] = y

    else:

        splits = train_test_split(X, y, train_size=train_size)
        X_train, X_test, y_train, y_test = splits

        data['X_train'] = X_train
        data['y_train'] = y_train

        if train_size+val_size == 1.0:
            data['X_val'] = X_test
            data['y_val'] = y_test

        elif train_size+test_size == 1.0:
            data['X_test'] = X_test
            data['y_test'] = y_test

        else:
            splits = train_test_split(
                                      X_test,
                                      y_test,
                                      test_size=test_size/(test_size+val_size),
                                      )
            X_val, X_test, y_val, y_test = splits
            data['X_val'] = X_val
            data['y_val'] = y_val
            data['X_test'] = X_test
            data['y_test'] = y_test

    return data


def load(names):
    '''
    Load data included with the package.

    Args:
        names (list): A list of data to load.

    Returns:
        Tuple[list, list]: A tuple of lists of features and target variables.
    '''

    Xs = []
    ys = []
    for name in names:

        if name == 'toy1':

            X = np.random.uniform(size=(1000, 3))
            y = 3+X[:, 0]+X[:, 1]**3+np.log(X[:, 2])

        elif name == 'toy2':

            X = np.random.uniform(-100, 50, size=(900, 3))
            y = 3+X[:, 0]+X[:, 1]**3+X[:, 2]

        elif name == 'friedman1':

            X = np.random.uniform(size=(500, 5))
            y = (
                 10*np.sin(np.pi*X[:, 0]*X[:, 1])
                 + 20*(X[:, 2]-0.5)**2
                 + 10*X[:, 3]
                 + 5*X[:, 4]
                 )

        else:
            path = os.path.join(data_path, f'{name}.csv')
            df = pd.read_csv(path)

            y = df['y'].values
            X = df.drop('y', axis=1).values

        Xs.append(X)
        ys.append(y)

    return Xs, ys

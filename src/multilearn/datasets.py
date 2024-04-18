from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import pkg_resources
import os

data_path = pkg_resources.resource_filename('multilearn', 'data')


def splitter(X, y, names=None, train_size=1.0, val_size=0.0, test_size=0.0):

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


def toy(points=[1000, 900, 500]):

    X1 = np.random.uniform(size=(points[0], 3))
    y1 = 3+X1[:, 0]+X1[:, 1]**3+np.log(X1[:, 2])

    X2 = np.random.uniform(size=(points[1], 3))
    y2 = 3+X2[:, 0]+X2[:, 1]**3+X2[:, 2]

    X3 = np.random.uniform(size=(points[2], 5))
    y3 = (
          10*np.sin(np.pi*X3[:, 0]*X3[:, 1])
          + 20*(X3[:, 2]-0.5)**2
          + 10*X3[:, 3]
          + 5*X3[:, 4]
          )

    X = [X1, X2, X3]
    y = [y1, y2, y3]

    return X, y


def load(names):

    Xs = []
    ys = []
    for name in names:
        path = os.path.join(data_path, f'{name}.csv')
        df = pd.read_csv(path)

        y = df['y'].values
        X = df.drop('y', axis=1).values

        Xs.append(X)
        ys.append(y)

    return Xs, ys

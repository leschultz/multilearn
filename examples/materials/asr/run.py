from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from models import MultiNet
from utils import train

import pandas as pd


def load(path):

    df = pd.read_csv(path)

    y = df['y'].values
    X = df.drop('y', axis=1).values

    return X, y

def loader(paths):

    Xs = []
    ys = []
    for i in paths:

        X, y = load(i)

        Xs.append(X)
        ys.append(y)

    return Xs, ys


def main():

    save_dir = 'outputs'
    lr = 1e-4
    batch_size = 32
    n_epochs = 1000

    # Data
    
    dtrain = loader([
                     '../../data/outputs/splits/train_asr.csv',
                     #'../../data/outputs/splits/train_opband.csv',
                     #'../../data/outputs/splits/train_stability.csv',
                     ])

    X_train, y_train = dtrain

    dval = loader([
                   '../../data/outputs/splits/val_asr.csv',
                   #'../../data/outputs/splits/val_opband.csv',
                   #'../../data/outputs/splits/val_stability.csv',
                   ])

    X_val, y_val = dval

    dtest = loader([
                    '../../data/outputs/splits/test_asr.csv',
                    #'../../data/outputs/splits/test_opband.csv',
                    #'../../data/outputs/splits/test_stability.csv',
                    ])

    X_test, y_test = dtest

    n_datasets = len(X_train)
    scalers = [StandardScaler() for _ in range(n_datasets)]
    losses = [nn.L1Loss() for _ in range(n_datasets)]

    model = MultiNet(tasks=n_datasets, input_arch={500: 1})
    optimizer = optim.Adam

    out = train(
                model,
                optimizer,
                losses,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                scalers=scalers,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                save_dir=save_dir,
                )

    print(out)


if __name__ == '__main__':
    main()

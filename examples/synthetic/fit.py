from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multilearn import datasets, models, utils
from torch import optim, nn

import pandas as pd


def main():

    save_dir = 'outputs'
    lr = 1e-4
    batch_size = 32
    n_epochs = 1000
    tasks = ['data1', 'data2', 'data3']

    # Data
    X, y = datasets.toy()
    data = datasets.splitter(X, y, tasks, train_size=1)

    for k, v in data.items():
        data[k]['scaler'] = StandardScaler()
        data[k]['loss'] = nn.L1Loss()

    n_datasets = len(data)

    model = models.MultiNet(tasks=tasks, input_arch={500: 1})
    optimizer = optim.Adam

    out = utils.train(
                      model,
                      optimizer,
                      data,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      lr=lr,
                      save_dir=save_dir,
                      )

    print(out)


if __name__ == '__main__':
    main()

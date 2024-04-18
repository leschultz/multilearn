from sklearn.preprocessing import StandardScaler
from multilearn import datasets, models, utils
from torch import optim, nn


def main():

    save_dir = 'outputs'
    lr = 1e-4
    batch_size = 32
    n_epochs = 1000
    tasks = ['asr']

    # Data
    X, y = datasets.load(tasks)
    data = datasets.splitter(X, y, tasks, train_size=1)

    for k, v in data.items():
        data[k]['scaler'] = StandardScaler()
        data[k]['loss'] = nn.L1Loss()

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

    print(out['df_loss'])


if __name__ == '__main__':
    main()

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from multilearn import models, utils
from torch import optim, nn


def main():

    save_dir = 'outputs'
    lr = 1e-4
    batch_size = 32
    n_epochs = 200
    train_size = 0.8  # Traning fraction
    val_size = 1.0-train_size  # Validation fraction
    print_n = n_epochs//10

    # Combine data to load
    tasks = ['asr', 'opband', 'stability']
    locations = [f'../../sample_data/{i}.csv' for i in tasks]

    # Load data in dictionary (make sure to keep order for loading items)
    data = utils.load(
                      locations,
                      names=tasks,  # User defined name
                      targets=['y']*len(tasks),  # Target names
                      )

    # Scalers and loss corresponding to loaded Xs and ys
    for key, value in data.items():
        value['scaler'] = StandardScaler()
        value['loss'] = nn.L1Loss()

    # A single model that combines nodes during training
    model = models.MultiNet(
                            tasks=tasks,
                            input_arch={100: 1},
                            mid_arch={100: 1, 50: 1},
                            out_arch={50: 1, 10: 1}
                            )

    # The optimizer for the NN model
    optimizer = optim.Adam

    # Do CV to assess
    utils.cv(
             data,
             model,
             optimizer,
             RepeatedKFold(n_repeats=1),
             train_size=train_size,
             val_size=val_size,
             save_dir=save_dir,
             lr=lr,
             batch_size=batch_size,
             n_epochs=n_epochs,
             print_n=print_n,
             )

    # Save one model to all data
    model = utils.full_fit(
                           data,
                           model,
                           optimizer,
                           train_size=train_size,
                           val_size=val_size,
                           save_dir=save_dir,
                           lr=lr,
                           batch_size=batch_size,
                           n_epochs=n_epochs,
                           print_n=print_n,
                           )

    name = tasks[1]
    X_inference = data[name]['X']  # Data with dropped columns

    print(f'Model used for predicting {name}')
    print(model.predict(X_inference, name))


if __name__ == '__main__':
    main()

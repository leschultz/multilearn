from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from multilearn import models, utils
from torch import optim, nn

import pandas as pd
import numpy as np

import unittest
import shutil
import os


class ml_test(unittest.TestCase):

    def test_ml(self):

        save_dir = 'outputs'
        lr = 1e-4
        batch_size = 32
        n_epochs = 100
        patience = 10  # Learning loop patience
        train_size = 0.8  # Traning fraction
        val_size = 1.0-train_size  # Validation fraction
        print_n = n_epochs//10

        # Local data 1
        local1 = './local1.csv'

        # Create local data 1
        X1 = np.random.uniform(size=(500, 1))
        y1 = 10*np.sin(np.pi*X1[:, 0])

        # Save local data 1
        df = pd.DataFrame(X1)
        df['target_1'] = y1
        df.to_csv(local1, index=False)

        # Local data 2
        local2 = './local2.csv'

        # Create local data 2
        X2 = np.random.uniform(size=(50, 7))
        y2 = (
              10*np.sin(np.pi*X2[:, 0]*X2[:, 1])
              + 20*(X2[:, 2]-0.5)**2
              + 10*X2[:, 3]
              + 5*X2[:, 4]
              )

        # Save local data 2
        df = pd.DataFrame(X2)
        df['target_2'] = y2
        df.to_csv(local2, index=False)

        # Combine data to load
        locations = [local1, local2]

        # Load data in dictionary (make sure to keep order for loading items)
        tasks = ['name1', 'name2']
        data = utils.load(
                          locations,
                          names=tasks,  # User defined name
                          targets=['target_1', 'target_2'],  # Target names
                          drops=[None, ['5', '6']],  # Columns to drop
                          )

        # Clean generated csv file
        [os.remove(i) for i in locations]

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
                 patience=patience,
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
                               patience=patience,
                               print_n=print_n,
                               )

        name = tasks[1]
        X_inference = data[name]['X']  # Data with dropped columns

        print(f'Model used for predicting {name}')
        print(model.predict(X_inference, name))

        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()

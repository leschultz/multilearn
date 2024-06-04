from lightning.pytorch.utilities.combined_loader import CombinedLoader
from multilearn.utils import to_tensor, loader
from torch import nn

import pandas as pd
import torch
import copy


class model_wrapper:

    def __init__(self, model, device=None):

        # Chose defalut device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        else:
            self.device = device

        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)

    def fit(self, data, optimizer, **kwargs):

        print_n = kwargs['print_n']
        n_epochs = kwargs['n_epochs']
        batch_size = kwargs['batch_size']
        lr = kwargs['lr']
        patience = kwargs['patience']

        data = copy.deepcopy(data)  # Avoids editing original data
        optimizer = optimizer(self.model.parameters(), lr=lr)

        # Fit scalers
        self.scalers = {}  # Copy them for prediction method
        for key, value in data.items():
            self.scalers[key] = copy.deepcopy(value['scaler'])
            self.scalers[key].fit(value['X_train'])

        # Apply transforms when needed
        data_train = {}
        for key, value in data.items():
            for k, v in value.items():

                # Transform features
                if ('X_' in k) and ('scaler' in value.keys()):
                    value[k] = self.scalers[key].transform(value[k])

                # Convert to tensor
                if ('X_' in k) or ('y_' in k):
                    value[k] = to_tensor(value[k], self.device)

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

            self.model.train()

            for batch, _, _ in data_train:

                loss = 0.0
                for indx in data.keys():

                    if batch[indx] is None:
                        continue

                    X = batch[indx][0]
                    y = batch[indx][1]

                    p = self.model(X, indx)
                    i = data[indx]['loss'](p, y)

                    if 'weight' in data[indx].keys():
                        i *= data[indx]['weight']

                    loss += i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.model.eval()

                all_loss = 0.0
                for indx in data.keys():
                    y = data[indx]['y_train']
                    p = self.model(data[indx]['X_train'], indx)
                    loss = data[indx]['loss'](p, y).item()

                    split = 'train'
                    d = (epoch, loss, indx, split)
                    df_loss.append(d)

                    if 'y_val' in data[indx].keys():

                        y = data[indx]['y_val']
                        p = self.model(data[indx]['X_val'], indx)
                        loss = data[indx]['loss'](p, y).item()

                        split = 'val'
                        d = (epoch, loss, indx, split)
                        df_loss.append(d)

                        all_loss += loss

                    else:
                        all_loss += loss

            # Early stopping
            if all_loss < best_loss:
                best_model = copy.deepcopy(self.model)
                best_loss = all_loss
                no_improv = 0

            else:
                no_improv += 1

            if no_improv >= patience:
                break

            if epoch % print_n == 0:
                print(f'Epoch {epoch}/{n_epochs}: {split} loss {loss:.2f}')

        if patience is not None:
            self.model = best_model

        # Loss curve
        columns = ['epoch', 'loss', 'data', 'split']
        df_loss = pd.DataFrame(df_loss, columns=columns)
        self.df_loss = df_loss

        return self.df_loss

    def predict(self, X, prop, device=None):

        X = self.scalers[prop].transform(X)

        if device is None:
            X = to_tensor(X, self.device)
            model = self.model.to(self.device)
        else:
            X = to_tensor(X, device)
            model = self.model.to(device)

        y_pred = model(X, prop)
        y_pred = y_pred.cpu().detach().view(-1).numpy()

        return y_pred


class MultiNet(nn.Module):
    '''
    A general model for building multi-target learning NNs.
    Each separation of layers is symmetric across input datasets.
    '''

    def __init__(
                 self,
                 input_arch={},
                 mid_arch={64: 1, 32: 1},
                 out_arch={},
                 tasks=[0],
                 ):

        super(MultiNet, self).__init__()

        def make_layers(arch, is_out=False):

            hidden = nn.ModuleList()
            for neurons, layers in arch.items():
                for i in range(layers):
                    hidden.append(nn.LazyLinear(neurons))
                    hidden.append(nn.LeakyReLU())

            if is_out:
                hidden.append(nn.LazyLinear(1))

            hidden = nn.Sequential(*hidden)

            return hidden

        def separate(arch, tasks, is_out=False):

            separate = nn.ModuleDict()
            for t in tasks:
                i = make_layers(arch, is_out)
                separate[t] = i

            return separate

        self.input = separate(input_arch, tasks)
        self.mid = make_layers(mid_arch)
        self.out = separate(out_arch, tasks, True)

    def forward(self, x, prop=None):
        '''
        Use a model to predict.

        Args:
            x (nn.tensor): The features.
            prop: The property to predict.

        Returns:
            torch.FloatTensor: The predicted target value.
        '''

        for i in self.input[prop]:
            x = i(x)

        for i in self.mid:
            x = i(x)

        for i in self.out[prop]:
            x = i(x)

        return x

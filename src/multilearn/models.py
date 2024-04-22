from torch import nn


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

    def forward(self, x, prop):
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

from torch import nn


class MultiNet(nn.Module):

    def __init__(
                 self,
                 input_arch={},
                 mid_arch={1024: 1, 32: 1, 16: 1},
                 out_arch={},
                 tasks=1,
                 ):

        super(MultiNet, self).__init__()

        def make_layers(arch, is_out=False):

            hidden = nn.ModuleList()
            for neurons, layers in arch.items():
                for i in range(layers):
                    hidden.append(nn.LazyLinear(neurons))
                    hidden.append(nn.ReLU())

            if is_out:
                hidden.append(nn.LazyLinear(1))

            hidden = nn.Sequential(*hidden)

            return hidden

        def separate(arch, task_iterator, is_out=False):

            separate = nn.ModuleList()
            for i in task_iterator:
                i = make_layers(arch, is_out)
                separate.append(i)

            return separate

        task_iterator = range(tasks)
        self.input = separate(input_arch, task_iterator)
        self.mid = make_layers(mid_arch)
        self.out = separate(out_arch, task_iterator, True)

    def forward(self, x, prop):

        for i in self.input[prop]:
            x = i(x)

        for i in self.mid:
            x = i(x)

        for i in self.out[prop]:
            x = i(x)

        return x

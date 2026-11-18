from torch import nn

class BarrierFuncNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BarrierFuncNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        #self.shortcut = nn.Linear(input_dim, 1)

    def forward(self, x):
        #return self.shortcut(x) + self.layer(x)
        return self.layer(x)
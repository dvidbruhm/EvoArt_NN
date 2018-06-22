import torch
import torch.nn as nn

import utils
import params

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(params.nb_params, 100)
        self.layer2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, params.channels),
            nn.Tanh()
        )

        self.apply(utils.init_weights)
    
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        return output
    
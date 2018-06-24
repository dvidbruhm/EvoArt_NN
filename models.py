import torch
import torch.nn as nn

import utils
import params

class Net(nn.Module):
    def __init__(self, depth, nb_neurons_per_layer):
        super(Net, self).__init__()

        activation = nn.Tanh()

        self.whole_net = nn.Sequential(nn.Linear(params.nb_input_params + 1, nb_neurons_per_layer))

        for i in range(depth):
            self.whole_net.add_module("hidden_layer_" + str(i), nn.Linear(nb_neurons_per_layer, nb_neurons_per_layer))
            self.whole_net.add_module("activation_" + str(i), activation)
        
        self.whole_net.add_module("output_layer", nn.Linear(nb_neurons_per_layer, params.channels))
        self.whole_net.add_module("output_activation", activation)

        self.apply(utils.init_weights)
    
    def forward(self, input):
        output = self.whole_net(input)
        return output
    
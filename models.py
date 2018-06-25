import torch
import torch.nn as nn

import utils
import settings

class Net(nn.Module):
    def __init__(self, depth, nb_neurons_per_layer):
        super(Net, self).__init__()

        self.activation = nn.Tanh()

        self.input_layer = nn.Sequential(nn.Linear(settings.nb_input_params + 1, nb_neurons_per_layer))

        self.hidden_layers = nn.Sequential()

        for i in range(depth):
            self.hidden_layers.add_module("hidden_layer_" + str(i), nn.Linear(nb_neurons_per_layer, nb_neurons_per_layer))
            #self.hidden_layers.add_module("activation_" + str(i), self.activation)
        
        self.output_layer = nn.Sequential(
            nn.Linear(nb_neurons_per_layer, settings.channels),
            self.activation
        )

        self.apply(utils.init_weights)
    
    def forward(self, input):
        output = self.input_layer(input)
        for layer in self.hidden_layers:
            output = layer(output)
            output = self.activation(output)
        output = self.output_layer(output)
        return output
    
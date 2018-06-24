import numpy as np
import torch.nn as nn

import settings
import models

def create_random_individual():

    latent_vec = np.repeat(np.random.normal(0, settings.latent_std, 1), settings.image_size ** 2).reshape(settings.image_size ** 2, -1)

    individual = Individual(
        models.Net(1, settings.nb_neuron_per_layer).to(settings.device),
        latent_vec,
        np.random.randint(1, settings.max_coord_scale),
        nn.Tanh()
    )

    #individual.print_attributes()

    return individual

def create_population(size=25):
    population = []

    for i in range(size):
        population.append(create_random_individual())
    
    return population


class Individual:
    def __init__(self, net, latent_vector, scale, activation_function):
        self.net = net
        self.latent_vector = latent_vector
        self.scale = scale
        self.activation_function = activation_function
    
    def print_attributes(self):
        print("[latent=", self.latent_vector[0], ", scale=", self.scale, ", activation=", self.activation_function, "]")

    def mutate(self, probablity):
        pass
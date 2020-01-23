import numpy as np
import torch.nn as nn
import copy
import random

import settings
import models


def next_generation(selected_individuals, size):

    next_population = []

    ### Crossover : create children based on selected individals of previous generation
    #               for each combination of parents

    for i in range(len(selected_individuals)):
        for j in range(i+1, len(selected_individuals)):
            if len(next_population) < size:
                parent1 = selected_individuals[i]
                parent2 = selected_individuals[j]

                child1, child2 = crossover(parent1, parent2)
                next_population.append(child1)
                if len(next_population) < size:
                    next_population.append(child2)

    ### Mutation : each new individual has a chance of mutating
    mutation(next_population, settings.mutation_probability)

    next_population = fill_population(next_population, size)

    return next_population


def create_random_individual():

    latent_vec = np.repeat(np.random.normal(0, settings.latent_std, 1), settings.image_size ** 2).reshape(settings.image_size ** 2, -1)

    individual = Individual(
        models.Net(random.randint(1, settings.init_nb_layers), settings.nb_neuron_per_layer).to(settings.device),
        latent_vec,
        np.random.randint(1, settings.max_coord_scale),
        nn.Tanh()
    )

    return individual


def fill_population(population, size):

    if size > len(population):
        for i in range(size - len(population)):
            population.append(create_random_individual())

    return population


def mutation(population, probablity):

    for individual in population:
        individual.mutate(probablity)


def crossover(parent1, parent2):

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    max_crossover = np.minimum(len(parent1.net.hidden_layers), len(parent2.net.hidden_layers))
    crossover_point = np.random.randint(1, max_crossover + 1) # +1 because max value is exclusive

    # Convert the nn.Sequantial object to list so it supports slicing
    layers1 = list(child1.net.hidden_layers)
    layers2 = list(child2.net.hidden_layers)

    layers1[:crossover_point], layers2[:crossover_point] = layers2[:crossover_point], layers1[:crossover_point]

    # Reconvert list to nn.Sequential object
    child1.net.hidden_layers = nn.Sequential(*list(layers1))
    child2.net.hidden_layers = nn.Sequential(*list(layers2))

    return child1, child2

class Individual:
    def __init__(self, net, latent_vector, scale, activation_function):
        self.net = net
        self.latent_vector = latent_vector
        self.scale = scale
        self.activation_function = activation_function


    def print_attributes(self):
        print("[latent=", self.latent_vector[0], ", scale=", self.scale, ", activation=", self.activation_function, "]")


    def mutate(self, probablity):
        if np.random.uniform() < probablity:
            layers = list(self.net.hidden_layers)
            rand_index = np.random.randint(0, len(layers))

            if np.random.randint(0, 4) <= 2 or len(layers) <= 1:
                ### Add layer
                layers.insert(rand_index, nn.Linear(settings.nb_neuron_per_layer, settings.nb_neuron_per_layer))
            else:
                ### Remove layer
                del layers[rand_index]

            self.net.hidden_layers = nn.Sequential(*layers)
            self.net.to(settings.device)

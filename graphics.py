import numpy as np
import torch
import pygame
from timeit import default_timer as timer

import settings
import utils

def draw_population(screen, population):
    images = []
    for individual in population:
        images.append(individual_to_image(individual))
    
    grid_size = int(np.sqrt(len(images)))
    for i in range(grid_size):
        for j in range(grid_size):
            image = images[i*grid_size + j]
            x = settings.padding+(i * (settings.image_resolution + settings.padding))
            y = settings.padding+(j * (settings.image_resolution + settings.padding))
            
            draw_image(screen, image, x, y)


def individual_to_image(individual):
    if settings.show_generation_time:
        print("--------------------------------------")
        print("Generating image...")
        start = timer()

    inputs = utils.create_grid(settings.image_size, settings.image_size, individual.scale)
    inputs = inputs[:settings.nb_input_params] + (individual.latent_vector,)
    inputs = np.concatenate(inputs, axis=1)

    with torch.no_grad():
        inputs = torch.Tensor(inputs).to(settings.device)
        output = individual.net(inputs)

        image = []

        for channel in range(settings.channels):
            chan_data = output[:, channel].cpu().numpy()
            chan_data = utils.normalize(chan_data, 255)
            image.append(chan_data.reshape(settings.image_size, settings.image_size))

        image = np.dstack(image)

        if settings.channels == 1:
            image = np.repeat(image, 3, axis=2)

    if settings.show_generation_time:
        end = timer()
        print("Generation took : ", end - start, " seconds.")
        print("--------------------------------------")

    return image


def net_to_image(net, latent_vector):
    if settings.show_generation_time:
        print("--------------------------------------")
        print("Generating image...")
        start = timer()

    inputs = utils.create_grid(settings.image_size, settings.image_size, settings.max_coord_scale)

    inputs = inputs[:settings.nb_input_params] + (latent_vector,)

    input = np.concatenate(inputs, axis=1)

    with torch.no_grad():
        input = torch.Tensor(input).to(settings.device)
        output = net(input)

        image = []

        for channel in range(settings.channels):
            chan_data = output[:, channel].cpu().numpy()
            chan_data = utils.normalize(chan_data, 255)
            image.append(chan_data.reshape(settings.image_size, settings.image_size))

        image = np.dstack(image)

        if settings.channels == 1:
            image = np.repeat(image, 3, axis=2)

    if settings.show_generation_time:
        end = timer()
        print("Generation took : ", end - start, " seconds.")
        print("--------------------------------------")

    return image

def draw_image(screen, image, x = 0, y = 0, resolution = settings.image_resolution):
    surface = pygame.surfarray.make_surface(image)
    surface = pygame.transform.scale(surface, (resolution, resolution))
    screen.blit(surface, (x, y))
    pygame.display.flip()


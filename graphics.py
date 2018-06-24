import numpy as np
import torch
import pygame
from timeit import default_timer as timer

import params
import utils

def net_to_image(net, latent_vector):
    if params.show_time:
        print("--------------------------------------")
        print("Generating image...")
        start = timer()

    inputs = utils.create_grid(params.image_size, params.image_size, params.coord_scale)
    #print(inputs[0].shape)
    #print(latent_vector.shape)
    inputs = inputs[:params.nb_input_params] + (latent_vector,)
    #print(inputs.shape)

    input = np.concatenate(inputs, axis=1)

    with torch.no_grad():
        input = torch.Tensor(input).to(params.device)
        output = net(input)

        image = []

        for channel in range(params.channels):
            chan_data = output[:, channel].cpu().numpy()
            chan_data = utils.normalize(chan_data, 255)
            image.append(chan_data.reshape(params.image_size, params.image_size))

        image = np.dstack(image)

        if params.channels == 1:
            image = np.repeat(image, 3, axis=2)

    if params.show_time:
        end = timer()
        print("Generation took : ", end - start, " seconds.")
        print("--------------------------------------")

    return image

def generate_image_slow(net):
    if params.show_time:
        print("--------------------------------------")
        print("Generating image...")
        start = timer()

    image = np.zeros((params.image_size, params.image_size, 3))

    origin = (params.image_size * params.coord_scale) / 2

    for x in range(0, params.image_size):
        for y in range(0, params.image_size):
            with torch.no_grad():
                x_norm = (x * params.coord_scale - origin) / (params.image_size/2)
                y_norm = (y * params.coord_scale - origin) / (params.image_size/2)
                r = np.sqrt((x_norm) ** 2 + (y_norm) ** 2)
                
                input = torch.Tensor([x_norm, y_norm, r, params.current_time]).to(params.device)
                rgb = net(input)

                image[x, y] = rgb

    image = utils.normalize(image, 255)

    if params.show_time:
        end = timer()
        print("Generation took : ", end - start, " seconds.")
        print("--------------------------------------")

    return image

def draw_image(screen, image, x = 0, y = 0, resolution = params.image_resolution):
    surface = pygame.surfarray.make_surface(image)
    surface = pygame.transform.scale(surface, (resolution, resolution))
    screen.blit(surface, (x, y))
    pygame.display.flip()


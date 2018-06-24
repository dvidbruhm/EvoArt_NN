import pygame
import torch

import utils
import params
import models
import graphics

import numpy as np

if __name__ == "__main__":

    #torch.manual_seed(1111)

    print("Device : ", params.device)

    pygame.init()

    screen = pygame.display.set_mode(params.win_size)

    sign = 1.0
    image = None
    current_net = None


    while True:
        events = pygame.event.get()
        utils.handle_inputs(events)


        if params.generate_next:
            #current_net = models.Net().to(params.device)
            # temp
            latent = np.random.normal(0, 100, 1)
            latent_vec = np.repeat(latent, params.image_size ** 2).reshape(params.image_size ** 2, -1)

            for i in range(1):

                for j in range(1):
                    params.weight_std = np.random.randint(1, 20)
                    params.coord_scale = np.random.randint(1, 10)
                    nb_hidden_layers = np.random.randint(1, 8)
                    nb_neurons = np.random.randint(1, 100)

                    current_net = models.Net(5, 100).to(params.device)

                    image = graphics.net_to_image(current_net, latent_vec)

                    graphics.draw_image(screen, image, x=j*params.image_resolution, y=i*params.image_resolution)

                    print("[scale=", params.coord_scale, ", std=", params.weight_std, ", ", nb_hidden_layers, ", ", nb_neurons, ", ", latent, "]")


        if params.animate and image is not None:

            if params.current_time > params.time_scale:
                sign = -1.0
            elif params.current_time < -params.time_scale:
                sign = 1.0

            params.current_time += sign * params.time_step

            image = graphics.net_to_image(current_net, latent_vec)
            graphics.draw_image(screen, image)
        
        pygame.time.wait(10)

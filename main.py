import pygame
import torch
import numpy as np

import utils
import settings
import models
import graphics
import genetics

if __name__ == "__main__":

    #torch.manual_seed(1111)

    print("Device : ", settings.device)

    pygame.init()

    screen = pygame.display.set_mode(settings.window_size)

    sign = 1.0
    image = None
    current_individual = None
    population = None


    while True:
        events = pygame.event.get()
        utils.handle_inputs(events)


        if settings.generate_next:
            population = genetics.create_population(25)
            graphics.draw_population(screen, population)
            """
            for i in range(5):
                counter = 0

                for j in range(5):
                    settings.weights_std = np.random.randint(1, 20)
                    counter += 1
                    settings.coord_scale = np.random.randint(1, 10)
                    nb_hidden_layers = np.random.randint(2, 10)
                    #nb_neurons = np.random.randint(1, 100)

                    current_net = models.Net(1, 100).to(settings.device)

                    latent = np.random.normal(0, 100, 1)
                    latent_vec = np.repeat(latent, settings.image_size ** 2).reshape(settings.image_size ** 2, -1)
                    image = graphics.net_to_image(current_net, latent_vec)

                    graphics.draw_image(screen, image, x=j*settings.image_resolution, y=i*settings.image_resolution)

                    #print("[scale=", params.coord_scale, ", std=", params.weight_std, ", ", nb_hidden_layers, ", ", nb_neurons, ", ", latent, "]")
                    current_individual = genetics.create_random_individual()
                    image = graphics.individual_to_image(current_individual)
                    graphics.draw_image(screen, image, x=j*settings.image_resolution, y=i*settings.image_resolution)
            """

        if settings.animate:

            if settings.current_time > settings.time_scale:
                sign = -1.0
            elif settings.current_time < -settings.time_scale:
                sign = 1.0

            settings.current_time += sign * settings.time_step

            graphics.draw_population(screen, population)

            #image = graphics.individual_to_image(current_individual)
            #graphics.draw_image(screen, image)
        
        pygame.time.wait(10)

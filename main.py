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

        if settings.animate:

            if settings.current_time > settings.time_scale:
                sign = -1.0
            elif settings.current_time < -settings.time_scale:
                sign = 1.0

            settings.current_time += sign * settings.time_step

            graphics.draw_population(screen, population)
        
        pygame.time.wait(10)

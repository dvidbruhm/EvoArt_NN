
import pygame
import torch

import utils
import params
import models
import graphics

if __name__ == "__main__":

    print("Device : ", params.device)

    pygame.init()

    screen = pygame.display.set_mode(params.win_size)

    sign = 1.0
    image = None
    net = None

    while True:
        events = pygame.event.get()
        utils.handle_inputs(events)


        if params.generate_next:
            #net = models.Net().to(params.device)

            counter = 0
            for i in range(5):
                for j in range(5):
                    net = models.Net().to(params.device)
                    counter += 1
                    params.weight_std = counter
                    print(params.weight_std)

                    image = graphics.net_to_image(net)

                    graphics.draw_image(screen, image, x=j*params.image_resolution, y=i*params.image_resolution)


        if params.animate and image is not None:

            if params.current_time > params.time_scale:
                sign = -1.0
            elif params.current_time < -params.time_scale:
                sign = 1.0

            params.current_time += sign * params.time_step

            image = graphics.net_to_image(net)
            graphics.draw_image(screen, image)
        
        pygame.time.wait(10)

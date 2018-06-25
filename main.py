import pygame
import torch
import numpy as np
import sys

import utils
import settings
import models
import graphics
import genetics

if __name__ == "__main__":

    #torch.manual_seed(1111)
    #np.random.seed(1111)

    ### set cpu or gpu fr pytorch
    print("Device : ", settings.device)

    ### pygame init
    pygame.init()
    screen = pygame.display.set_mode(settings.window_size)

    ### init
    sign = 1.0
    grid_offset = (0, 100)
    generate_next = False
    animate = False
    population = genetics.fill_population([], settings.population_size)
    graphics.draw_population(screen, population, offset=grid_offset)

    selected_indices = []

    while True:
        events = pygame.event.get()
        #utils.handle_inputs(events)

        for event in events:
            if event.type == pygame.QUIT: 
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    generate_next = True
                    break
                if event.key == pygame.K_SPACE:
                    animate = not animate
                    break
            if event.type == pygame.MOUSEBUTTONDOWN and (pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]):
            
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[0] > grid_offset[0] + settings.padding / 2 and mouse_pos[0] < settings.window_size[0] - settings.padding / 2 and  \
                   mouse_pos[1] > grid_offset[1] + settings.padding / 2 and mouse_pos[1] < settings.window_size[1] - settings.padding / 2:
                
                    index = utils.get_index(mouse_pos, grid_offset)

                    if index in selected_indices:
                        selected_indices.remove(index)
                    else:
                        selected_indices.append(index)

                    print(selected_indices)

        if generate_next:

            population = genetics.next_generation(population[0:5], settings.population_size)
            graphics.draw_population(screen, population, offset=grid_offset)

            generate_next = False

        if animate:

            if settings.current_time > settings.time_scale:
                sign = -1.0
            elif settings.current_time < -settings.time_scale:
                sign = 1.0

            settings.current_time += sign * settings.time_step

            graphics.draw_population(screen, population, offset=grid_offset)
        
        pygame.time.wait(10)

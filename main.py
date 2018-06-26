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
    state = utils.State()
    population = genetics.fill_population([], settings.population_size)
    images = graphics.population_to_images(population)
    graphics.draw_images(screen, images, offset=settings.grid_offset)

    while True:
        events = pygame.event.get()
        utils.handle_inputs(events, state)

        # Draw selection highlight if the selection changed
        if state.selection_changed:

            graphics.draw_selection(screen, state.selected_indices, settings.grid_offset)
            graphics.draw_images(screen, images, offset=settings.grid_offset)

            state.selection_changed = False

        # Generate next population based on user selection
        if state.generate_next:
            
            ### clear screen before drawing
            screen.fill((0,0,0))

            selected_individuals = [population[i] for i in state.selected_indices]
            population = genetics.next_generation(selected_individuals, settings.population_size)
            images = graphics.population_to_images(population)
            graphics.draw_images(screen, images, offset=settings.grid_offset)

            settings.current_time = 1.0
            state.reset()

        # Animate the images
        if state.animate:

            if settings.current_time > settings.time_scale:
                sign = -1.0
            elif settings.current_time < -settings.time_scale:
                sign = 1.0

            settings.current_time += sign * settings.time_step

            images = graphics.population_to_images(population)
            graphics.draw_images(screen, images, offset=settings.grid_offset)
        
        pygame.time.wait(10)

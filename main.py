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
    myfont = pygame.font.SysFont('Comic Sans MS', 18)

    ### buttons
    next_button = graphics.Button(screen, settings.window_size[0] - 100 - 50, settings.grid_offset[1]/2 - 25, 100, 50, "Next", myfont, (0, 255, 0), (0, 255, 0), toggle=False)
    next_button.draw()

    animate_button = graphics.Button(screen, 50, settings.grid_offset[1]/2 - 25, 100, 50, "Animate", myfont, (255, 0, 0), (0, 0, 255), toggle=True)
    animate_button.draw()

    save_button = graphics.Button(screen, settings.window_size[0]/2 - 75, settings.grid_offset[1]/2 - 25, 150, 50, "Save selected", myfont, (0, 255, 0), (0, 255, 0), toggle=False)
    save_button.draw()

    ### init
    sign = 1.0
    state = utils.State()
    population = genetics.fill_population([], settings.population_size)
    images = graphics.population_to_images(population)
    graphics.draw_images(screen, images, offset=settings.grid_offset)
    pygame.display.set_caption("Evolutionary CPPN")

    while True:
        events = pygame.event.get()
        utils.handle_inputs(events, state, next_button, animate_button, save_button)
        
        # Draw selection highlight if the selection changed
        if state.selection_changed:

            graphics.draw_selection(screen, state.selected_indices, settings.grid_offset)
            graphics.draw_images(screen, images, offset=settings.grid_offset)

            state.selection_changed = False

        # Generate next population based on user selection
        if state.generate_next:
            
            ### clear screen before drawing
            utils.fill_image_grid(screen)
            next_button.draw()

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
        
        # Save selection
        if state.save:
            selected_individuals = [population[i] for i in state.selected_indices]
            selected_images = [graphics.individual_to_image(ind, image_size=settings.saved_image_resolution) for ind in selected_individuals]
            utils.save_images(selected_images)
            state.save = False

        pygame.display.flip()
        pygame.time.wait(10)

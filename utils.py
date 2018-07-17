import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import pygame
import sys
import os

import settings

class State:
    selected_indices = []
    generate_next = False
    animate = False
    selection_changed = False
    save = False

    def reset(self):
        self.selected_indices = []
        self.generate_next = False
        self.animate = False
        self.selection_changed = False
        self.save = False

def handle_inputs(events, state, next_button, animate_button, save_button):
    for event in events:
        if event.type == pygame.QUIT: 
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                state.generate_next = True
                break
            if event.key == pygame.K_SPACE:
                state.animate = not state.animate
                break
        if event.type == pygame.MOUSEBUTTONDOWN and (pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]):
        
            mouse_pos = pygame.mouse.get_pos()

            ### buttons
            if next_button.is_clicked(mouse_pos):
                next_button.click()
                state.generate_next = True

            if animate_button.is_clicked(mouse_pos):
                animate_button.click()
                state.animate = not state.animate

            if save_button.is_clicked(mouse_pos):
                save_button.click()
                state.save = True

            ### Image selection
            if mouse_pos[0] > settings.grid_offset[0] + settings.padding / 2 and mouse_pos[0] < settings.window_size[0] - settings.padding / 2 and  \
                mouse_pos[1] > settings.grid_offset[1] + settings.padding / 2 and mouse_pos[1] < settings.window_size[1] - settings.padding / 2:
            
                index = get_index(mouse_pos, settings.grid_offset)

                if index in state.selected_indices:
                    state.selected_indices.remove(index)
                else:
                    state.selected_indices.append(index)

                state.selection_changed = True

def get_index(mouse_pos, offset):
    i = int((mouse_pos[0] - settings.padding / 2 - offset[0]) / (settings.image_resolution+settings.padding))
    j = int((mouse_pos[1] - settings.padding / 2 - offset[1]) / (settings.image_resolution+settings.padding))
    index = j*settings.grid_size + i
    return index

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(settings.weights_mean, np.sqrt(np.random.randint(1, settings.max_weights_std)/m.in_features))
        m.bias.data.zero_()

def create_grid(x_dim, y_dim, scale = 1.0):
    N = np.mean((x_dim, y_dim))
    x = np.linspace(- x_dim / N * scale, x_dim / N * scale, x_dim)
    y = np.linspace(- y_dim / N * scale, y_dim / N * scale, y_dim)

    X, Y = np.meshgrid(x, y)

    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    r = np.sqrt(x ** 2 + y ** 2)
    time = np.ones_like(x) * settings.current_time
    cosx = np.cos(x)
    cosy = np.cos(y)
    sum = x + y
    difference = x - y


    if settings.horizontal_symetry:
        x = x ** 2
        sum = sum ** 2
        difference = difference ** 2
    if settings.vertical_symetry:
        y = y ** 2
        if not settings.horizontal_symetry:
            sum = sum ** 2
            difference = difference ** 2


    return x, y, r, time, cosx, cosy, sum, difference

def normalize(data, max_value=1):
    min = np.min(data)
    max = np.max(data)
    if max - min < 0.01:
        return data
    return (data - min)/(max - min) * max_value

def fill_image_grid(screen, color=(0, 0, 0)):
    
    screen.fill((0,0,0), rect=(
        settings.grid_offset[0], 
        settings.grid_offset[1], 
        settings.window_size[0]-settings.grid_offset[0], 
        settings.window_size[1]-settings.grid_offset[1]
        )
    )

def save_images(images, path="saved/"):
    os.makedirs(path, exist_ok=True)
    existing_images = os.listdir(path)
    current_index = 0
    if len(existing_images) > 0:
        current_index = int(sorted(existing_images)[-1][:-4]) # get index of last saved image
    
    for image in images:
        current_index += 1
        image_data = torch.Tensor(normalize(image.transpose()))
        save_image(image_data, path + str(current_index) + ".png")
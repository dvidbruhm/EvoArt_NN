import torch
import torch.nn as nn
import numpy as np
import pygame
import sys

import settings

def handle_inputs(events):
    for event in events:
        if event.type == pygame.QUIT: 
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                settings.generate_next = True
                return
            if event.key == pygame.K_SPACE:
                settings.animate = not settings.animate
                return
        if event.type == pygame.MOUSEBUTTONDOWN and (pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]):
        
            mouse_pos = pygame.mouse.get_pos()
    
    settings.generate_next = False

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
    theta = np.arctan((y + np.max(y) + 1) / (x + np.max(x) + 1))    # weird additions to remove zero divide and to have theta of first quadrant
    time = np.ones_like(x) * settings.current_time
    cosx = np.cos(x)
    cosy = np.cos(y)
    sum = x + y
    difference = x - y


    if settings.horizontal_symetry:
        x = x ** 2
    if settings.vertical_symetry:
        y = y ** 2

    return x, y, r, theta, time, cosx, cosy, sum, difference

def normalize(data, max_value=1):
    min = np.min(data)
    max = np.max(data)
    if max - min < 0.01:
        return data
    return (data - min)/(max - min) * max_value

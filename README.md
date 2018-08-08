# EvoArt
Very simple program to create evolutionary art using CPPNs and a genetic algorithm. 

# Requirements
- [python 3](https://www.python.org/)
- [pygame](https://www.pygame.org/)
- [pytorch](https://www.pytorch.org)

Note: the program will run much faster with a cuda compatible GPU, but a CPU is still usable.

# Usage
Clone or download the repository, then run main.py:
```
python3 main.py
```
### Generating images

A window will appear with a grid of generated images. To start creating art, select some images by clicking on them, then click on the __next__ button (top right) to generate a new population of images based on your selected images from the previous population. Repeat this process until you like the images.

At any moment the __animate__ button (top left) can be clicked to animate the grid of images currently displayed.

Note: If no images are selected and the __next__ button is clicked, a new random population will be generated.

### Saving images

When you are happy with the results, you can select the images you want to save (same as before, by clicking on them), and click on the __Save selected__ button (top middle). The saved images will go to a folder named "saved/" and will be in the ".png" format. The resolution of the saved images can be changed in the settings.py file (see [Configuration](https://github.com/dvidbruhm/EvoArt_NN/blob/master/README.md#configuration)).

# Configuration

The settings.py file contains all the configuration and can be changed to obtain different results. For example, the `population_size` parameter can be decreased to speed up the generation on CPU, or the `saved_image_resolution` parameter can be modified to change the output images resolution. Symetry can be added to the images by changing either `horizontal_symetry` or `vertical_symetry`, or both. 

Here are the defaults:

```
### genetic algorithm settings

mutation_probability = 0.5
population_size = 25

### graphics settings

window_size = (620, 720)
image_size = 100
image_resolution = 100
channels = 1                # 1 for black/white or 3 for colors

padding = 20
grid_size = 5
grid_offset = (0, 100)

max_coord_scale = 3

saved_image_resolution = 1000

### animation settings

time_scale = 10
time_step = 0.2
current_time = 1.0

### model settings

nb_neuron_per_layer = 15
init_nb_layers = 2

nb_input_params = 6
horizontal_symetry = False
vertical_symetry = False

weights_mean = 0.0
max_weights_std = 20
latent_std = 10

### pytorch settings
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### debug settings

show_generation_time = False
```

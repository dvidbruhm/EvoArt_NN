### genetic algorithm settings

mutation_probability = 0.5
population_size = 25

### graphics settings

window_size = (620, 720)
image_size = 100
image_resolution = 100
channels = 3                # 1 for black/white or 3 for colors

max_coord_scale = 10

padding = 20
grid_size = 5
grid_offset = (0, 100)

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

#device = "cpu"
device = "cuda:0"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### debug settings

show_generation_time = False

### global variables (do not change)

generate_next = False
animate = False
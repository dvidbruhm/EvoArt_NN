image_size = 300
image_resolution = 300
channels = 3
nb_input_params = 6

coord_scale = 5

grid_size = 5
win_size = (300, 300)

time_scale = 10
time_step = 0.2
current_time = 1.0

weight_mean = 0.0
weight_std = 50

#device = "cpu"
device = "cuda:0"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

show_time = False

generate_next = False
animate = False
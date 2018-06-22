image_size = 50
image_resolution = 50
channels = 1
nb_params = 2

coord_scale = 1

grid_size = 5
win_size = (250, 250)

time_scale = 10
time_step = 1
current_time = 1.0

weight_mean = 0.0
weight_std = 1

device = "cpu"
#device = "cuda:0"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

show_time = False

generate_next = False
animate = False
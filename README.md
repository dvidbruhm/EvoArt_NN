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

A window will appear with a grid of generated images. To start creating art, just select the images you prefer by clicking on them, then click on the __next__ button (top right) to generate a new population of images based on your selected images from the previous population. Repeat this process until you like the images.

Note: If no images are selected and the __next__ button is clicked, a new random population will be generated.

### Saving images

When you are happy with the results, you can select the images you want to save (same as before, by clicking on them), and click on the __Save selected__ button. The saved images will go to a folder named "saved/" and will be in the ".png" format. The resolution of the saved images can be changed in the settings.py file (see ).

# Configuration


import numpy as np
from PIL import Image

# Converts image to array of pixels
def read_image(file_name):
    return np.asarray(Image.open(file_name).convert('L'))

# Saves array of pixels to a png file
def write_image(image, path):
    png_image = Image.fromarray(np.array(image), 'L')
    png_image.save(path)

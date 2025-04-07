import PIL
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def image_grid2(imgs, rows, cols):
    assert len(imgs) == rows*cols
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def mima(array, MIN, MAX):
    min_val = MIN
    max_val = MAX
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def download_image(url):
    image = PIL.Image.open(url)
    image = np.asarray(image)
    image = mima(image.astype("float32"), 0, 65536)
    image = np.asarray([image, image, image])
    image = image[np.newaxis, :]
    image = torch.from_numpy(image)
    return image

def imshow(img2):
    plt.figure(figsize=(15, 6))
    plt.axis('off')
    plt.imshow(img2)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re


def load_images(image_paths):
    return [Image.open(path) for path in image_paths]


def save_gif(images, filename, duration=100):
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)


def natural_sort_key(s):
    # 使用正則表示式將字串分割為數字和非數字部分
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


# 自動切換到 main.py 所在目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))
directory = "filling_results"
image_paths = os.listdir(directory)

for index, filename in enumerate(image_paths):
    image_paths[index] = os.path.join(directory, filename)

image_paths = sorted(image_paths, key=natural_sort_key)
print(image_paths)
images = load_images(image_paths)

# 保存為 GIF 動畫
save_gif(images, 'filling_animation.gif')

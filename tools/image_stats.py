import os

import PIL
import seaborn as sns
from matplotlib import pyplot as plt


def plot_dimensions_scatterplot(images_dir):
    widths, heights = [], []
    for directory in os.listdir(images_dir):
        for img_file in os.listdir(os.path.join(images_dir, directory)):
            img = PIL.Image.open(os.path.join(images_dir, directory, img_file))
            width, height = img.size
            widths.append(width)
            heights.append(height)

    sns.jointplot(widths, heights)
    plt.show()


if __name__ == "__main__":
    images_dir = '../data/low-resolution-cropped'
    plot_dimensions_scatterplot(images_dir)

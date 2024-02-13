from matplotlib import pyplot as plt
import numpy as np

def show_grid(images, nb_cols = None, output_file = None):
    plt.figure()
    if nb_cols is None:
        nb_cols = min(10, len(images))
    nb_lines = len(images) // nb_cols + (len(images) % nb_cols > 0)

    fig, axes = plt.subplots(nb_lines, nb_cols, figsize=(1.5*nb_cols, 1.5*nb_lines))
    axes = axes.flatten()

    for i in range(len(images)):
        im = np.swapaxes(images[i, :, :, :], 0, 2)
        im = (im + 1) / 2
        im = np.maximum(im, np.zeros(im.shape))
        im = np.minimum(im, np.ones(im.shape))

        ax = axes[i]
        ax.imshow(im)

    for i in range(nb_lines * nb_cols):
        axes[i].axis('off')

    plt.tight_layout()

    if output_file is not None:
        print(output_file)
        plt.savefig(output_file)

    plt.show()
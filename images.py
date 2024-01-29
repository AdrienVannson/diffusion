from matplotlib import pyplot as plt
import numpy as np

def show_grid(images):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(25):
        ax = axes[i]
        ax.imshow(np.swapaxes(images[i, :, :, :], 0, 2))
        ax.axis('off')

    plt.tight_layout()
    plt.show()
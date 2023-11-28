import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import vars as v

def visualize(pred, label, save=True):
    pred, label = pred.cpu().detach().view(28 ,28).numpy(), label.cpu().detach().view(28, 28).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(label, cmap="gray")
    ax[0].set_title('Original Image')

    ax[1].imshow(pred, cmap="gray")
    ax[1].set_title('Predicted Reconstruction')

    if save:
        os.makedirs('../comparisons', exist_ok=True)
        plt.savefig(f'../comparisons/comparison_{v.current_epoch}.png')
    plt.close(fig)

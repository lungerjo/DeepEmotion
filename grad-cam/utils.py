import numpy as np
import matplotlib.pyplot as plt
from pylab import *


def visualize_3d_solid(volume, label, threshold=0.5):
    """
    Visualize a 3D binary solid using Matplotlib
    """
    binary_volume = (volume > threshold)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(binary_volume, facecolors='blue', edgecolors='k', alpha=0.6)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(label.item())

    plt.show()


def visualize_3d_heatmap(heatmap):
    """
    Visualize the 3d heatmap as a 3d volume.
    """
    data = heatmap.detach().numpy()

    x, y, z = np.indices(data.shape)
    values = data.flatten()
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111,projection='3d')
    norm = plt.Normalize(values.min(), values.max())
    yg = ax.scatter(x, y, z, c=values, marker='s', s=200, cmap="Greens_r")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


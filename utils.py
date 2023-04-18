"""
Various utility functions for the project.
Authors: Brock Dyer, Dade Wood
CSCI 736
"""

from math import floor


def compute_kernel_sizes(s, num_layers):
    """
    Compute the kernel sizes for all layers of the network.

    Params
    ------
    s: The size of the neighbor window.
    num_layers: Number of layers in the network.

    Return
    ------
    A list of the kernel sizes for the network layers.
    """
    kernels = []
    size = s
    for i in range(num_layers-1):
        k = floor(size / 2) + 1
        size = size - (k - 1)
        kernels.append(k)
    # The last layer in the network sets the kernel to the previous layer's
    # output size
    kernels.append(size)
    return kernels

"""
Various utility functions for the project.
Authors: Brock Dyer, Dade Wood
CSCI 736
"""

import torch


def compute_kernel_sizes(s, num_layers):
    """
    Compute the kernel sizes for all layers of the network.

    Params
    ------
    s
        The size of the neighbor window.
    num_layers
        Number of layers in the network.

    Return
    ------
    A list of the kernel sizes for the network layers.
    """
    kernels = []
    size = s
    for i in range(num_layers-1):
        k = (size // 2) + 1
        size = size - (k - 1)
        kernels.append(k)
    # The last layer in the network sets the kernel to the previous layer's
    # output size
    kernels.append(size)
    return kernels


def extract_sequential_data(x):
    """
    Extract the sequential sensing data.

    Parameters
    ----------
    x
        The input tensor.
    """
    return torch.transpose(torch.flatten(x, start_dim=2, end_dim=3),
                           dim0=1, dim1=2)


def extract_spectral_data(x, window_size):
    """
    Extract the spectral sensing data.

    Parameters
    ----------
    x
        The input tensor.
    """
    return x[:, :, window_size // 2, window_size // 2]

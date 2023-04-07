##################################################
# Various utility functions for the project.
# Authors: Brock Dyer
# CSCI 736
##################################################

from math import floor

def compute_kernel_size(s, layer_index):
    """
    Compute the kernel size for a given network layer.

    Params
    ------
    s : The size of the neighbor window.
    layer_index : The index of the layer. Uses 0-based indexing.

    Return
    ------
    The dimension/kernel size of the given network layer.
    """
    if layer_index <= 0:
        return floor(s / 2) + 1

    return floor(compute_kernel_size(s, layer_index - 1) / 2) + 1

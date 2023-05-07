############################################################
# Loss functions for the model
# Author: Brock Dyer
# CSCI 736
############################################################

import torch.nn as nn
import torch

MSE = nn.MSELoss()


def reconstruction_loss(input_vector, predicted_vector):
    """
    Compute the reconstruction loss term between the decoded pixel vector and
    the original pixel vector supplied as input.

    Params
    ------
    input_vector: The original input pixel vector from the image.
    predicted_vector: The decoded pixel vector from the decoder network.

    Returns
    -------
    The mean-squared error (reconstruction loss) between the two vectors.
    """
    return MSE(predicted_vector, input_vector)


def kl_loss(mean_vector, std_vector):
    """
    Compute the kl-divergence loss term between the latent distribution and
    the normal distribution.

    Params
    ------
    mean_vector: The mean vector which parameterizes the latent distribution.
    std_vector: The std vector which parameterizes the latent space.

    Returns
    -------
    The kl-divergence between the latent distribution and the normal
    distribution.
    """

    # kl = ((torch.square(std_vector) + torch.square(mean_vector) -
    #        torch.log(torch.square(std_vector)) - 1) / 2).sum()
    # print(mean_vector)
    # print(std_vector)
    kl = (std_vector ** 2 + mean_vector ** 2
          - torch.log(std_vector) - 0.5).sum()
    return kl


def homology_loss(xls, xss):
    """
    Compute the homology loss term. This is a KL-Divergence between the local
    sensing features and the sequential sensing features.

    Params
    ------
    xls: The encoding of the local sensing features.
    xss: The encoding of the sequential sensing features.

    Returns
    -------
    The computed homology loss.
    """
    xls = xls + 1e-12
    xss = xss + 1e-12

    sum_term1 = xls * torch.log(xls / xss)
    sum_term2 = xss * torch.log(xss / xls)

    # print("SUM TERM1", sum_term1)
    # print("SUM TERM2", sum_term2)
    sum_term = sum_term1 + sum_term2
    sum_result = torch.sum(sum_term)
    return sum_result / 2

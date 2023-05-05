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

def kl_loss(mean_vector, variance_vector):
    """
    Compute the kl-divergence loss term between the latent distribution and 
    the normal distribution.

    Params
    ------
    mean_vector: The mean vector which parameterizes the latent distribution.
    variance_vector: The variance vector which parameterizes the latent space.

    Returns
    -------
    The kl-divergence between the latent distribution and the normal
    distribution.
    """
    term1 = -torch.log(torch.square(variance_vector))
    term2 = torch.square(mean_vector)
    term3 = torch.square(variance_vector)

    numerator = term1 + term2 + term3 - 1
    return numerator / 2

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

    sum_term1 = xls * torch.log(xls / xss)
    sum_term2 = xss * torch.log(xss / xls)
    sum_term = sum_term1 + sum_term2
    sum_result = torch.sum(sum_term)
    return sum_result / 2

def loss(input_vector, predicted_vector, mean_vector, variance_vector, xls, xss):
    """
    Compute the loss for a given sample run through the network.

    Params
    ------
    input_vector: The original input pixel vector from the image.
    predicted_vector: The decoded pixel vector from the decoder network.
    mean_vector: The mean vector which parameterizes the latent distribution.
    variance_vector: The variance vector which parameterizes the latent space.
    xls: The encoding of the local sensing features.
    xss: The encoding of the sequential sensing features.

    Returns
    -------
    The loss value for the sample.
    """

    reconstruction_term = reconstruction_loss(input_vector, predicted_vector)
    kl_term = kl_loss(mean_vector, variance_vector)
    homology_term = homology_loss(xls, xss)

    return reconstruction_term + kl_term + homology_term
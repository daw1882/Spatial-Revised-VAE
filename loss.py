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

    sum_term1 = xls * torch.log(xls / xss)
    sum_term2 = xss * torch.log(xss / xls)

    # print("SUM TERM1", sum_term1)
    # print("SUM TERM2", sum_term2)
    sum_term = sum_term1 + sum_term2
    sum_result = torch.sum(sum_term)
    return sum_result / 2


def VAE_loss(input_vector, predicted_vector, xls, xss):
    """
    Compute the loss for a given sample run through the network.

    Params
    ------
    input_vector: The original input pixel vector from the image.
    predicted_vector: The decoded pixel vector from the decoder network.
    mean_vector: The mean vector which parameterizes the latent distribution.
    std_vector: The std vector which parameterizes the latent space.
    xls: The encoding of the local sensing features.
    xss: The encoding of the sequential sensing features.

    Returns
    -------
    The loss value for the sample.
    """
    # print("LOSS VALUES:")
    # print("Input Vector:", input_vector, input_vector.size(), torch.count_nonzero(input_vector), torch.any(input_vector < 0))
    # print("Predicted Vector:", predicted_vector, predicted_vector.size(), torch.count_nonzero(predicted_vector), torch.any(predicted_vector < 0))
    # print("Mean Vector:", mean_vector.size())
    # print("Standard Dev Vector:", std_vector.size())
    # print("xls Vector:", xls, xls.size())
    # print("xss Vector:", xss, xss.size())

    # inputs = torch.clone(input_vector)
    # targets = torch.clone(predicted_vector)

    reconstruction_term = reconstruction_loss(input_vector, predicted_vector)
    # kl_term = kl_loss(mean_vector, std_vector)
    homology_term = homology_loss(xls, xss)
    # print("reconstruction loss:", reconstruction_term, reconstruction_term.size())
    # print("kl loss:", kl_term, kl_term.size())
    # print("homology loss:", homology_term)

    # result = reconstruction_term + kl_term + homology_term
    # print("Result:", result, result.size())
    return reconstruction_term, homology_term

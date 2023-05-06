############################################################
# Loss functions for the model
# Author: Brock Dyer
# CSCI 736
############################################################

import torch.nn as nn
import torch

MSE = nn.MSELoss()
KL_loss = nn.KLDivLoss(reduction="batchmean")

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
# def kl_loss(input_vector, predicted_vector):
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
    term1 = -torch.log(torch.square(std_vector))
    term2 = torch.square(mean_vector)
    term3 = torch.square(std_vector)

    numerator = term1 + term2 + term3 - 1
    return numerator / 2

    # func = nn.Sigmoid()
    # input_vector = func(input_vector)

    # predicted_vector = torch.log(predicted_vector)
    # input_vector = torch.abs(input_vector)
    # print("HOW IS THIS HAPPENING", torch.any(input_vector < 0))
    # print("TESTING", torch.log(input_vector), predicted_vector)
    # return KL_loss(predicted_vector, input_vector)


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

    print("SUM TERM1", sum_term1)
    print("SUM TERM2", sum_term2)
    sum_term = sum_term1 + sum_term2
    sum_result = torch.sum(sum_term)
    return sum_result / 2


def VAE_loss(input_vector, predicted_vector, mean_vector, std_vector, xls, xss):
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
    print("LOSS VALUES:")
    print("Input Vector:", input_vector, input_vector.size(), torch.count_nonzero(input_vector), torch.any(input_vector < 0))
    print("Predicted Vector:", predicted_vector, predicted_vector.size(), torch.count_nonzero(predicted_vector), torch.any(predicted_vector < 0))
    print("Mean Vector:", mean_vector.size())
    print("Standard Dev Vector:", std_vector.size())
    print("xls Vector:", xls, xls.size())
    print("xss Vector:", xss, xss.size())

    # inputs = torch.clone(input_vector)
    # targets = torch.clone(predicted_vector)

    reconstruction_term = reconstruction_loss(input_vector, predicted_vector)
    kl_term = kl_loss(mean_vector, std_vector)
    homology_term = homology_loss(xls, xss)
    print("reconstruction loss:", reconstruction_term, reconstruction_term.size())
    print("kl loss:", kl_term, kl_term.size())
    print("homology loss:", homology_term)

    result = reconstruction_term + kl_term + homology_term
    # print("Result:", result, result.size())
    return result
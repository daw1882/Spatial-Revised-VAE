"""
Local sensing using the proposed CNN model.

Author: Dade Wood
CSCI 736
"""

from collections import OrderedDict

import torch.nn as nn

from spectral_spatial_vae.utils import compute_kernel_sizes


class LocalSensingNet(nn.Module):

    def __setup_layer(self, s, ld, spectral_bands, index):
        """
        Setup one of the CNN layers.

        Parameters
        ----------
        s: The neighborhood window size.
        ld: The size of the final latent dimension, used for number of filters.
        spectral_bands: The number of spectral bands.
        index: The layer index, starting at 0.
        """
        input_size = spectral_bands if index == 0 else ld
        output_size = ld if index < (self.num_layers-1) else ld//4
        kernel_size = self.kernel_sizes[index]

        layer_name = f"Layer {index}"
        cnn = nn.Conv2d(input_size, output_size, kernel_size)
        return layer_name, cnn

    def __init__(self, s, ld, spectral_bands, num_layers=3):
        """
        The CNN network for local sensing.

        The input of this network is sxsxN neighborhood region around the pizel
        vector of size Nx1.

        The output of this network is an (ld/4)x1 vector containing the local
        spacial features used to modify the mean and the std. dev. of the VAE.

        Parameters
        ----------
        s
            The size of the neighborhood window. Paper suggests a size of 11.
        ld
            The size of the latent dimension or representation. This is also used
            for the size of the hidden layers
        spectral_bands
            The number of hyperspectral bands.
        num_layers
            The number of cnn layers. Paper suggests 3 as a default.
        """
        super(LocalSensingNet, self).__init__()
        self.num_layers = num_layers
        self.kernel_sizes = compute_kernel_sizes(s, num_layers)

        cnn_layer_list = []
        for index in range(num_layers):
            cnn_layer_list.append(
                self.__setup_layer(s, ld, spectral_bands, index)
            )

        self.cnn_layers = nn.Sequential(OrderedDict(cnn_layer_list))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the network.

        Parameters
        ----------
        x
            The input data.

        Returns
        -------
            A (ld/4)x1 vector of local spatial features.
        """
        return self.activation(self.cnn_layers(x))

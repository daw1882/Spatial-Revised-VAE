############################################################
# Sequential feature encoding using VAE
# Author: Brock Dyer
# CSCI 736
############################################################

import torch.nn as nn

class SpectralEncoder(nn.Module):
    """
    The spectral variational encoder. Takes a pixel vector as input, and outputs
    the mean and variance. The mean is a tensor (1, ld / 2), and the variance
    is a tensor (1, ld). The mean is revised by the spatial encodings in a later
    step.

    Parameters
    ----------
    spectral_bands : The number of spectral bands.
    ld : The dimension of the latent vector.
        This should always be a multiple of 4.
    layers : The number of dense stacked layers in the encoder.
        The last layer splits into the mean and variance layers.
    """

    def __init__(self, spectral_bands, ld, layers) -> None:
        """
        Initialize the network. 

        Parameters
        ----------
        spectral_bands : The number of spectral bands.
        ld : The dimension of the latent vector.
        layers : The number of dense stacked layers in the encoder.
        """
        super(SpectralEncoder, self).__init__()

        hidden_size = ld
        variance_layer_input_size = hidden_size
        mean_layer_input_size = hidden_size
        mean_layer_output_size = ld // 2

        self.layers = []

        if layers >= 2:
            # First layer is 
            self.layers.append(nn.Linear(spectral_bands, hidden_size))
            for _ in range(1, layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        else:
            mean_layer_input_size = spectral_bands
            variance_layer_input_size = spectral_bands

        self.mean_layer = nn.Linear(mean_layer_input_size, mean_layer_output_size)
        self.variance_layer = nn.Linear(variance_layer_input_size, ld)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : A pixel vector.

        Returns
        -------
        A tuple of the form (mean_vector, variance_vector) where the mean vector
        is the encoding of the mean with adjusted dimensions, and the variance
        vector is the encoding of the variance.
        """

        # Forward pass of stacked layers (n-1) layers.
        for layer in self.layers:
            x = self.activation(layer(x))

        # Compute mean vector
        mean_vector = self.activation(self.mean_layer(x))

        # Compute variance vector
        variance_vector = self.activation(self.variance_layer(x))
        
        return (mean_vector, variance_vector)
        



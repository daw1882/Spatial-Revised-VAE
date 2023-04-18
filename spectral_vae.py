############################################################
# Sequential feature encoding using VAE
# Author: Brock Dyer
# CSCI 736
############################################################

import torch
import torch.nn as nn
from sequential_sensing import SequentialSensingNet
from local_sensing import LocalSensingNet

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

    def __init__(self, spectral_bands, ld, layers=3) -> None:
        """
        Initialize the network. 

        Parameters
        ----------
        spectral_bands : The number of spectral bands.
        ld : The dimension of the latent vector.
        layers : The number of dense stacked layers in the encoder.
            Default is 3 layers.
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
        
        # TODO: Check if this is allowed, may need to convert this into a tensor.
        return (mean_vector, variance_vector)
        
class SpectralSpatialEncoder(nn.Module):
    """
    The combined encoder model. Uses the spatial features to modify the mean
    of the variational auto encoder.
    """

    def __init__(self, s, ld, spectral_bands, layers, ss_layers, ls_layers) -> None:
        """
        Initialize the module.

        Parameters
        ----------
        s : The neighborhood window size.
        ld : The number of dimensions in the latent space.
        spectral_bands : The number of spectral bands in the image.
        layers : The number of layers in the spatial encoder.
        ss_layers : The number of layers in the sequential sensing encoder.
        ls_layers : The number of layers in the local sensing encoder.
        """
        super(SpectralSpatialEncoder, self).__init__()

        self.spat_encoder_ss = SequentialSensingNet(s, ld, spectral_bands, ss_layers)
        self.spat_encoder_ls = LocalSensingNet(s, ld, spectral_bands, ls_layers)
        self.spec_encoder = SpectralEncoder(spectral_bands, ld, layers)

        self.neighbor_window_size = s
        self.latent_dimensions = ld
        self.spectral_bands = spectral_bands


    def extract_sequential_data(self, x):
        """
        Extract the sequential sensing data.

        Parameters
        ----------
        x : The input image tensor.
        """
        # TODO: Implement
        pass

    def extract_local_data(self, x):
        """
        Extract the local sensing data.

        Parameters
        ----------
        x : The input image tensor.
        """
        # TODO: Implement
        pass

    def split_mean_variance(self, spec_encoder_result) -> tuple:
        """
        Return the mean vector and variance vector from the spectral encoding
        output.

        Parameters
        ----------
        spec_encoding_result: The output from spectral encoding network.

        Returns
        -------
        A tuple containing the mean tensor in the first position and the
        variance tensor in the second.
        """

        # For now this is a tuple, so just return it.
        return spec_encoder_result

    def forward(self, x):
        """
        Perform the forward pass.

        Parameters
        ----------
        x : The input image tensor.
        """

        # Split the data for each of the encoder stacks.
        seq_sensing_data = self.extract_sequential_data(x)
        loc_sensing_data = self.extract_local_data(x)
        spectral_encoding_data = x

        # Pass data to each encoder
        
        # Output shape for spatial encodings is (1, ld // 4)
        xss = self.spat_encoder_ss(seq_sensing_data)
        xls = self.spat_encoder_ls(loc_sensing_data)

        # Output shape for spectral mean is (1, ld // 2)
        # Output shape for spectral variance is (1, ld)
        mv, vv = self.split_mean_variance(self.spec_encoder(spectral_encoding_data))

        # Revise the mean by concatenating the vectors.
        # Concatenation order is xls + mv + xss
        mv = torch.concat((xls, mv, xss), 1)

        # TODO: Verify that this is acceptable output format. May have to turn into a tensor.
        return (mv, vv)



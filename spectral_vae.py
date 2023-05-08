############################################################
# Sequential feature encoding using VAE
# Author: Brock Dyer
# CSCI 736
############################################################

import torch
import torch.nn as nn
from sequential_sensing import SequentialSensingNet
from local_sensing import LocalSensingNet
import utils

from loss import homology_loss, kl_loss


class SpectralEncoder(nn.Module):
    """
    The spectral variational encoder. Takes a pixel vector as input, and outputs
    the mean and std. The mean is a tensor (1, ld / 2), and the std
    is a tensor (1, ld). The mean is revised by the spatial encodings in a later
    step.

    Parameters
    ----------
    spectral_bands : The number of spectral bands.
    ld : The dimension of the latent vector.
        This should always be a multiple of 4.
    layers : The number of dense stacked layers in the encoder.
        The last layer splits into the mean and std layers.
    """

    def __init__(self, spectral_bands, ld, device, layers=3) -> None:
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
        std_layer_input_size = hidden_size
        mean_layer_input_size = hidden_size
        mean_layer_output_size = ld // 2

        self.layers = []

        if layers >= 2:
            # First layer is 
            self.layers.append(nn.Linear(spectral_bands, hidden_size,
                                         device=device))
            for _ in range(1, layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size,
                                             device=device))
        else:
            mean_layer_input_size = spectral_bands
            std_layer_input_size = spectral_bands

        self.mean_layer = nn.Linear(mean_layer_input_size,
                                    mean_layer_output_size, device=device)
        self.std_layer = nn.Linear(std_layer_input_size, ld, device=device)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : A pixel vector.

        Returns
        -------
        A tuple of the form (mean_vector, std_vector) where the mean vector
        is the encoding of the mean with adjusted dimensions, and the std
        vector is the encoding of the std.
        """

        # Forward pass of stacked layers (n-1) layers.
        for layer in self.layers:
            x = layer(x)

        # Compute mean vector
        mean_vector = self.mean_layer(x)

        # Compute std vector
        std_vector = self.std_layer(x)

        return mean_vector, std_vector


def split_mean_std(spec_encoder_result) -> tuple:
    """
    Return the mean vector and std vector from the spectral encoding
    output.

    Parameters
    ----------
    spec_encoding_result: The output from spectral encoding network.

    Returns
    -------
    A tuple containing the mean tensor in the first position and the
    std tensor in the second.
    """

    # For now this is a tuple, so just return it.
    return spec_encoder_result


class SpectralSpatialEncoder(nn.Module):
    """
    The combined encoder model. Uses the spatial features to modify the mean
    of the variational auto encoder.
    """

    def __init__(self, s, ld, spectral_bands, layers, ss_layers, ls_layers,
                 device) -> None:
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
        self.device = device

        self.spat_encoder_ss = SequentialSensingNet(s, ld, spectral_bands, ss_layers)
        self.spat_encoder_ls = LocalSensingNet(s, ld, spectral_bands, ls_layers)
        self.spec_encoder = SpectralEncoder(spectral_bands, ld, device, layers)
        self.norm = torch.distributions.Normal(0, 1)

        self.neighbor_window_size = s
        self.latent_dimensions = ld
        self.spectral_bands = spectral_bands
        self.kl = 0
        self.homology = 0

    # Data Tensor: (#pixel vectors, SxS, N) (all at once)
    # Data Tensor: (SxS, N) one at a time.
    def forward(self, x):
        """
        Perform the forward pass.

        Parameters
        ----------
        x : The input tensor.
        """

        # Split the data for each of the encoder stacks.
        # Dimension of input x: (batch, s, s, N)

        # Dimension of output: (batch, s^2, N)
        seq_sensing_data = utils.extract_sequential_data(x)
        # Dimension of output: (batch, N, s, s)
        # loc_sensing_data = utils.extract_local_data(x)
        loc_sensing_data = x
        # Dimension of output: (batch, 1, N), the center pixel vector
        spectral_encoding_data = utils.extract_spectral_data(x, self.neighbor_window_size)

        # Pass data to each encoder
        
        # Output shape for spatial encodings is (batch, ld // 4)
        xss = torch.squeeze(self.spat_encoder_ss(seq_sensing_data))
        xls = torch.squeeze(self.spat_encoder_ls(loc_sensing_data))
        self.homology = homology_loss(xls, xss)

        # Output shape for spectral mean is (batch, ld // 2)
        # Output shape for spectral std is (batch, ld)
        mv, sv = split_mean_std(self.spec_encoder(spectral_encoding_data))

        # Revise the mean by concatenating the vectors.
        # Concatenation order is xls + xss + mv
        mv = torch.concat((xls, xss, mv), 1)
        # TODO: look into the exp
        sv = torch.exp(sv)

        # Re-parameterization trick
        z = mv + sv * self.norm.sample(mv.shape).to(self.device)

        # compute the KL divergence and store in the class
        self.kl = kl_loss(mv, sv)

        return z


class SpectralSpatialDecoder(nn.Module):

    def __init__(self, ld, spectral_bands, layers, device) -> None:
        """
        Initialize the module.

        Parameters
        ----------
        ld : The number of dimensions in the latent space.
        spectral_bands : The number of spectral bands.
        layers : The number of layers in the spatial encoder.
        """
        super(SpectralSpatialDecoder, self).__init__()

        hidden_size = ld
        output_size = spectral_bands

        self.ld = ld
        self.activation = nn.Sigmoid()
        self.layers = []

        if layers <= 1:
            self.layers.append(nn.Linear(ld, output_size, device=device))
            return

        self.layers.append(nn.Linear(ld, hidden_size, device=device))
        # Middle layers
        for _ in range(1, layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, device=device))

        self.layers.append(nn.Linear(hidden_size, output_size, device=device))

    def forward(self, x):
        """
        Perform forward pass.
        """

        x_hat = x

        for layer in self.layers:
            x_hat = layer(x_hat)

        return x_hat


class SpatialRevisedVAE(nn.Module):
    def __init__(self, s, ld, spectral_bands, device, layers=3, ss_layers=3,
                 ls_layers=3):
        super(SpatialRevisedVAE, self).__init__()
        self.spectral_bands = spectral_bands
        self.encoder = SpectralSpatialEncoder(s, ld, spectral_bands, layers,
                                              ss_layers, ls_layers, device)
        self.decoder = SpectralSpatialDecoder(ld, spectral_bands, layers,
                                              device)
        # self.mu = None
        # self.std = None

    def forward(self, x):
        z = self.encoder(x)
        # self.mu = z[0]
        # self.std = z[1]
        return self.decoder(z)


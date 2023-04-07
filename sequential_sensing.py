############################################################
# Sequential sensing using the proposed LSTM model.
# Author: Brock Dyer
# CSCI 736
############################################################

import torch
import torch.nn as nn

from collections import OrderedDict

from utils import compute_kernel_size

class SequentialSensingNet(nn.Module):
    """
    The LSTM network for sequential sensing.

    The input of this network is a sxsxN neighborhood region around the pixel
    vector which is Nx1.

    The output of this network is an (ld/4)x1 vector containing the sequential
    spacial features used to modify the mean and std. dev. of the VAE.

    Note that each layer of the LSTM is modeled similarly to a CNN, where each
    layer works on a smaller dimensional region.

    Hyperparameters
    ---------------
    s : The size of the neighborhood window. Paper suggests a size of 11.
    ld : The size of the latent representation. This is also used for the size
        of hidden layers.
    spectral_bands : The number of hyperspectral bands.
    lstm_layers : The number of stacked LSTM layers. Paper suggests 3 as a default.
    N : the batch size for the LSTM.
    """

    def __setup_layer(self, s, index, spectral_bands):
        """
        Setup one of the LSTM layers.

        Params
        ------
        s : The neighborhood window size.
        index: The layer index. Starting at 0.
        spectral_bands : The number of spectral bands.
        """
        input_size = spectral_bands if index == 0 else compute_kernel_size(s, index - 1)
        hidden_size = compute_kernel_size(s, index)

        layer_name = f"Layer {index}"
        lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        return (layer_name, lstm)

    def __init__(self, s, ld, spectral_bands, lstm_layers=3) -> None:
        super(SequentialSensingNet, self).__init__()
        
        lstm_layer_list = []
        for index in range(lstm_layers):
            lstm_layer_list.append(self.__setup_layer(s, index, spectral_bands))

        self.lstm_stack = nn.Sequential(OrderedDict(lstm_layer_list))
        self.avg_pooling = None # TODO: Setup average pooling layer.

    def forward(self, x):
        pass


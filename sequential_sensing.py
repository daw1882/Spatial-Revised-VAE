############################################################
# Sequential sensing using the proposed LSTM model.
# Author: Brock Dyer
# CSCI 736
############################################################

import torch
import torch.nn as nn

from collections import OrderedDict

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

    def __setup_layer(self, index, ld):
        """
        Setup one of the LSTM layers.

        Params
        ------
        s : The neighborhood window size.
        index: The layer index. Starting at 0.
        ld: The latent dimension.
        """
        input_size = ld
        hidden_size = ld

        layer_name = f"Layer {index}"
        lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        return (layer_name, lstm)

    def __init__(self, s, ld, lstm_layers=3) -> None:
        super(SequentialSensingNet, self).__init__()
        
        lstm_layer_list = []
        for index in range(lstm_layers - 1):
            lstm_layer_list.append(self.__setup_layer(index, ld))

        lstm_layer_list.append(self.__setup_layer(lstm_layers - 1, ld / 4))

        self.lstm_stack = nn.Sequential(OrderedDict(lstm_layer_list))
        self.avg_pooling = None # TODO: Setup average pooling layer.

    def forward(self, x):
        pass


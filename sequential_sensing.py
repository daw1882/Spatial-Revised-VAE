############################################################
# Sequential sensing using the proposed LSTM model.
# Author: Brock Dyer
# CSCI 736
############################################################

import torch
import torch.nn as nn


class ExtractLSTMOutput(nn.Module):
    def forward(self, x):
        output, _ = x
        return output


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
    """

    def __init__(self, s, ld, spectral_bands, lstm_layers=3) -> None:
        super(SequentialSensingNet, self).__init__()
        self.window_size = s

        # Setup stacked LSTM
        extractor = ExtractLSTMOutput()
        if lstm_layers == 1:
            stack = nn.LSTM(input_size=spectral_bands, hidden_size=ld // 4,
                                  num_layers=lstm_layers)
            self.lstm_stack = nn.Sequential(stack, extractor)
        else:
            # All hidden sizes are ld except for the final layer, which is output
            # as ld / 4.
            stack = nn.LSTM(input_size=spectral_bands, hidden_size=ld, num_layers=lstm_layers - 1)
            final = nn.LSTM(input_size=ld, hidden_size=ld // 4, num_layers=1)
            self.lstm_stack = nn.Sequential(stack, extractor, final, extractor)

        # Setup average pooling
        # Output from LSTM is tensor of shape (s x s, ld // 4)
        # Pooling needs to get (1, ld // 4)
        # Need to average along the s x s dimension.
        self.average_pooling = nn.AvgPool1d(s * s)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Perform the forward pass.

        Parameters
        ----------
        x : The input data.
            This should be a tensor of the shape (s^2, N) where s is the window
            size, and N is the number of spectral bands.
        """

        # Run forward pass through LSTM, outputs a tensor (s^2, ld)
        # print("Before lstm:", x.size())
        lstm_output = self.lstm_stack(x)
        # print("After lstm:", lstm_output.size())

        # Transpose output so that we can do average pooling
        lstm_output_t = torch.transpose(lstm_output, 1, 2)
        # print("After transpose:", lstm_output_t.size())

        # Perform average pooling
        pooled = self.average_pooling(lstm_output_t)
        # print("After pooling:", pooled.size())

        # Transpose back to (1, ld // 4)
        # result = torch.transpose(pooled, 1, 2)
        # print("After transpose:", result.size())
        result = self.activation(pooled)

        return result


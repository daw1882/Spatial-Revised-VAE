"""
Classes for creating a spectral image dataset for the Spatial-Spectral VAE in
the pytorch format.

Authors: Dade Wood, Zoe Lalena
CSCI 736
"""

import os

import cv2
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class SpectralImage:

    def __init__(self, image_path, data_type='dir', data_key=None):
        """
        Create a spectral image from a directory or .mat file.

        Parameters
        ----------
        image_path
            Path to the spectral image data.
        data_type
            Type of data to load. Options: {'dir', 'mat'}
        data_key
            Key to load the data in from a .mat file.
        """
        self.image = None
        self.width = None
        self.height = None
        self.data_key = data_key
        self.image_path = image_path
        self.spectral_bands = 0
        print("Loading in spectral image...")
        if data_type == 'mat':
            self.load_spectral_mat()
        else:
            self.band_paths = []
            image_names = os.listdir(image_path)
            for image in image_names:
                self.band_paths.append(image_path + "/" + image)
            self.load_spectral_dir()
        print("Spectral image loaded!")

    def load_spectral_mat(self):
        """Load a spectral image from a .mat file."""
        spec_img = scipy.io.loadmat(self.image_path)[self.data_key]
        spec_img = (spec_img - np.min(spec_img)) / (np.max(spec_img) - np.min(spec_img))
        self.width, self.height, self.spectral_bands = spec_img.shape
        self.image = np.transpose(spec_img, axes=(2, 1, 0))

    def load_spectral_dir(self):
        """Load a spectral image from a directory containing the bands."""
        self.spectral_bands = len(self.band_paths)
        for i, path in enumerate(self.band_paths):
            # Read in image in greyscale mode
            channel = cv2.imread(path, 0)
            channel = np.divide(channel, 255)
            channel = np.add(channel, 1e-12)
            channel = np.expand_dims(channel, axis=2)
            if self.image is None:
                self.height, self.width, _ = channel.shape
                self.image = channel
            else:
                self.image = np.append(self.image, channel, axis=2)
            print(f"\tBand {i+1} loaded.")
        self.image = np.transpose(self.image, axes=(2, 0, 1))

    def get_length(self, s):
        """
        Get length of the image given a window size.

        Parameters
        ----------
        s
            Size of the windows for the VAE.

        Returns
        -------
        Length of the image.
        """
        _, r, c = self.image.shape
        return self.image[0, 0:r-s, 0:c-s].size

    def get_window(self, idx, s):
        """
        Get a window of a given size from the spectral image.

        Parameters
        ----------
        idx
            Index of the center pixel for the window in the image.
        s
            Size of the square window to extract.

        Returns
        -------
        A square window extracted from the hyperspectral image.
        """
        row = (idx // (self.width - s))
        col = (idx % (self.width - s))
        return self.image[:, row: row+s, col: col+s]


class SpectralVAEDataset(Dataset):

    def __init__(self, spectral_im: SpectralImage, window_size, device):
        """
        Create a dataset of spectral pixel vectors from a spectral image.

        Parameters
        ----------
        spectral_im
            The spectral image to make the dataset from.
        window_size
            Size of the windows to extract surrounding the pixel
            vector.
        device
            The device to load the data on to.
        """
        self.spectral_im = spectral_im
        self.window_size = window_size
        self.device = device

    def __len__(self):
        return self.spectral_im.get_length(self.window_size)

    def __getitem__(self, idx):
        sample = self.spectral_im.get_window(idx, self.window_size)
        return torch.Tensor(sample).to(self.device)

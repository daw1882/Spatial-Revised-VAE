import os
import cv2
import numpy as np
import scipy

import torch
from torch.utils.data import Dataset, DataLoader


class SpectralImage:

    def __init__(self, image_path, data_type='dir', data_key=None):
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
        spec_img = scipy.io.loadmat(self.image_path)[self.data_key]
        spec_img = (spec_img - np.min(spec_img)) / (np.max(spec_img) - np.min(spec_img))
        self.width, self.height, self.spectral_bands = spec_img.shape
        self.image = np.transpose(spec_img, axes=(2, 1, 0))

    def load_spectral_dir(self):
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
        _, r, c = self.image.shape
        return self.image[0, 0:r-s, 0:c-s].size

    def get_window(self, idx, s):
        row = (idx // (self.width - s))
        col = (idx % (self.width - s))
        return self.image[:, row: row+s, col: col+s]


class SpectralVAEDataset(Dataset):

    def __init__(self, spectral_im: SpectralImage, window_size, device):
        self.spectral_im = spectral_im
        self.window_size = window_size
        self.device = device

    def __len__(self):
        return self.spectral_im.get_length(self.window_size)

    def __getitem__(self, idx):
        sample = self.spectral_im.get_window(idx, self.window_size)
        return torch.Tensor(sample).to(self.device)


if __name__ == '__main__':
    print("---------------------")
    print("DATASET TEST")
    im = SpectralImage("C:\\Users\\dade_\\NN_DATA\\testing")
    dataset = SpectralVAEDataset(im, 11, "cpu")

    print(len(dataset))
    for i in range(len(dataset)):
        x = dataset[i]
        print(i, x.size())

        if i == 3:
            break

    print("---------------------")
    print("DATALOADER TEST")
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())

        if i_batch == 3:
            break

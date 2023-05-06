import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class SpectralImage:

    def __init__(self, image_dir):
        self.image = None
        self.perm = None
        self.width = None
        self.height = None
        self.image_paths = []
        self.image_dir = image_dir
        image_names = os.listdir(image_dir)
        for image in image_names:
            self.image_paths.append(image_dir + "/" + image)
        self.spectral_bands = len(self.image_paths)
        self.load_spectral_image()

    def load_spectral_image(self):
        print("Loading in spectral image...")
        for path in self.image_paths:
            # Read in image in greyscale mode
            channel = cv2.imread(path, 0)
            channel = np.transpose(channel)
            channel = np.divide(channel, 255)
            channel = np.add(channel, 1e-12)
            channel = np.expand_dims(channel, axis=2)
            if self.image is None:
                self.width, self.height, _ = channel.shape
                self.image = channel
            else:
                self.image = np.append(self.image, channel, axis=2)
        print("Spectral image loaded!")

    def get_length(self, s):
        r, c, _ = self.image.shape
        return self.image[0:r-s, 0:c-s, 0].size

    def get_window(self, idx, s):
        row = (idx // (self.height - s))
        col = (idx % (self.height - s))
        return self.image[row: row+s, col: col+s, :]


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

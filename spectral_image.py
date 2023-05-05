import os
import cv2
import numpy as np


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
            self.image_paths.append(image_dir + "\\" + image)
        self.load_spectral_image()

    def load_spectral_image(self):
        print("Loading in spectral image...")
        for path in self.image_paths:
            # Read in image in greyscale mode
            channel = cv2.imread(path, 0)
            channel = np.transpose(channel)
            channel = np.expand_dims(channel, axis=2)
            if self.image is None:
                self.width, self.height, _ = channel.shape
                print(self.width)
                self.image = channel
            else:
                self.image = np.append(self.image, channel, axis=2)
            print(self.image.shape)
        print("Spectral image loaded!")

    def get_length(self, s):
        r, c, _ = self.image.shape
        return self.image[0:r-s, 0:c-s, 0].size

    def get_window(self, idx, s):
        row = (idx // (self.height - s))
        col = (idx % (self.height - s))
        return self.image[row: row+s, col: col+s, :]

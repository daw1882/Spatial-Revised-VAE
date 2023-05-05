import os
import cv2
import numpy as np


class SpectralImage:

    def __init__(self, image_dir):
        self.image = None
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
            channel = np.expand_dims(channel, axis=2)
            if self.image is None:
                self.image = channel
            else:
                self.image = np.append(self.image, channel, axis=2)
        print("Spectral image loaded!")


if __name__ == "__main__":
    im = SpectralImage("C:\\Users\\dade_\\NN_DATA\\testing")
    # im = SpectralImage("C:\\Users\\dade_\\NN_DATA\\318r\\TIFFs")
    im.load_spectral_image()

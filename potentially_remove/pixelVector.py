"""
Makes a version of image that is sXs where every element is a pixel vector containing
the spectral info for each band

Author: Zoe LaLena
Date: 4/8/2023
Course: NN & ML
"""
import os
import cv2
import numpy as np
def make_image_list(path):
    """
    Makes a list of all the images in a given directory
    :param path:  containing image files
    :return: list of images
    """
    image_names = os.listdir(path)
    # print(image_names)
    images = []
    for image in image_names:
        img = cv2.imread(path + "\\" + image, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images

def makePixelVectorMartix(folder_to_img):
    """
    Makes a version of image that is sXs where every element is a pixel vector containing
    the spectral info for each band
    :param folder_to_img: folder containing images for each band in msi
    :return: a sXsXn image where at each location in the sxs matrix is a pixel vector of a the spectral data at that point
    """
    images = make_image_list(folder_to_img)
    h, w= images[0].shape
    c = len(images)
    pixel_vect_mat = np.zeros((h,w, c))
    for col in range(0,w):
        for row in range(0,h):
            pixel_vector = []
            for chan in range(0,c):
                pixel_vector.append(images[chan][row,col])
            pixel_np = np.array(pixel_vector)
            pixel_vect_mat[row,col]  = pixel_np
    return pixel_vect_mat

def main():
    folder = "318r_selection/2266_2145"
    result = (makePixelVectorMartix(folder))
    print(result[0,0])

if __name__ == '__main__':
    main()
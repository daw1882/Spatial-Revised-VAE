"""
This code should actually do the thing this time

Author: Zoe LaLena
Date: 4/9/2023
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


def makePixelVectorMatrix(folder_to_img):
    """
    Makes a version of image that is sXs where every element is a pixel vector containing
    the spectral info for each band
    :param folder_to_img: folder containing images for each band in msi
    :return: a sXsXn image where at each location in the sxs matrix is a pixel vector of a the spectral data at that point
    """
    images = make_image_list(folder_to_img)
    h, w = images[0].shape
    c = len(images)
    pixel_vect_mat = np.zeros((h, w, c))
    for col in range(0, w):
        for row in range(0, h):
            pixel_vector = []
            for chan in range(0, c):
                pixel_vector.append(images[chan][row, col])
            pixel_np = np.array(pixel_vector)
            pixel_vect_mat[row, col] = pixel_np
    return pixel_vect_mat


def sBysRegions(pixel_vector_image, s=11):
    """
    Given an image where each location is a pixel vector of all the spectral info for that location, this function
    creates a list of sxs regions of info for each location in the image
    :param pixel_vector_image: Image with pixel vector of channel info at each location
    :param s: dimensions of regions
    :return: return vector of image regions, 1 for each loaction in the image
    """
    h, w, channels = pixel_vector_image.shape
    vector_of_regions = []
    for col in range(s, w - s):
        for row in range(s, h - s):
            vector_of_regions.append(pixel_vector_image[row: row + s, col:col + s,:])
    return vector_of_regions


def main():
    pixel_vector_img = makePixelVectorMatrix("D:\imageFiles\\318_all_aspects")
    vector_of_regions = sBysRegions(pixel_vector_img, 11)
    region1 = vector_of_regions[0]
    print(region1)
    region1_channel1 = region1[:,:,0]
    print(region1_channel1)
    print(region1.shape)
    cv2.imwrite("test.png", region1_channel1)


if __name__ == '__main__':
    main()

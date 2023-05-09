"""
Code to split multi-spectral image data into sxs chunks

Images are saved in to folders for each section containing all the bands for that chunk

Author: Zoe LaLena
Date: 4/8/2023
Course: NN & ML

"""

import os
import cv2


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


def divide_image(folder_of_images, save_location, s=11):
    """
    divided multi-spectral images in to smaller sections of size s
    :param folder_of_images: folder containing all the grey scale images for each band of a multi-spectral image
    :param save_location: where to save results to
    :param s: the dimensions of the smaller sections (sxs)
    :return: none
    """

    # get names of all image so we can read them in
    image_names = os.listdir(folder_of_images)

    # grab list of images
    images = make_image_list(folder_of_images)

    # make save loaction if it does not exist
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    w, h = images[0].shape

    # let figure out how many images in each dimension
    w_amount = int(w / s)
    h_amount = int(h / s)

    for r in range(0, w_amount):
        for c in range(0, h_amount):
            newpath = save_location + "/" + str(s * r) + "_" + str(s * c)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cur_img = 0
            for img in images:
                section = img[s * r:s * (r + 1), s * c:s * (c + 1)]
                cv2.imwrite(newpath + "/" + image_names[cur_img], section)
                cur_img = cur_img + 1


def main():
    divide_image("D:\Zoe\\318r\TIFFs", "318r")

if __name__ == '__main__':
    main()
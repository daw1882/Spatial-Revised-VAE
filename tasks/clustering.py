"""
Cluster an image for finding text in a reagant treated document.

Author: Dade Wood
CSCI 736
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm, colors
from sklearn.cluster import KMeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        help='Path to the csv data to load.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--output_path',
        help='Path to output the image to.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image_width',
        help='Width of the output image.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--image_height',
        help='Height of the output image.',
        required=True,
        type=int,
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv, header=None)

    X = df.to_numpy()

    print("--------------------")
    print("Training...")
    clf = KMeans(n_clusters=2, n_init=1000, max_iter=1000, tol=1e-8)
    labels = clf.fit_predict(X)
    print("Training done!")

    print("--------------------")
    print("Making image...")
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    pallette = ["#f8fa0d", "#3143b9"]
    cmap = colors.LinearSegmentedColormap.from_list(name="", colors=pallette)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    pixels = labels.reshape(args.image_height, args.image_width)
    pixels = np.uint8(mapper.to_rgba(pixels)*255)
    img = Image.fromarray(pixels)
    img.save(args.output_path)
    print("Image made!")

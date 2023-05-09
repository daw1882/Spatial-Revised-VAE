from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colormaps
from matplotlib import cm, colors


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
    # print(df.head(), df.shape)
    original_x = df.iloc[:, :-2].to_numpy()
    original_y = df.iloc[:, -1].to_numpy()
    # print(np.unique(original_y))

    # zero_x = original_x[original_y == 0]
    # zero_y = original_y[original_y == 0]
    X = original_x[original_y != 0]
    Y = original_y[original_y != 0]
    # print(X.shape, Y.shape)

    # zero_x_train, zero_x_test, zero_y_train, zero_y_test = train_test_split(
    #     zero_x, zero_y, test_size=0.95, random_state=42
    # )
    # print(zero_y_train.shape)
    # X = np.concatenate((X, zero_x_train), axis=0)
    # Y = np.concatenate((Y, zero_y_train), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.70, random_state=42, stratify=Y
    )

    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)
    # print(y_train[y_train == 0].shape)

    print("--------------------")
    print("Training...")
    # clf = svm.SVC(degree=9, gamma='auto')
    # clf = MLPClassifier()
    clf = KNeighborsClassifier(3)
    clf.fit(x_train, y_train)
    print("Training done!")
    print("--------------------")

    y_pred = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("--------------------")
    print("Making image...")
    print(X.shape)
    pixels = np.empty(original_y.shape)
    norm = colors.Normalize(vmin=0.0, vmax=16.0, clip=True)
    pallette = ["#3143b9", "#0262e0", "#0c74dc", "#1484d3", "#0898d1",
                "#05a6c6", "#15b0b4", "#37b89d", "#64be85", "#91be72",
                "#b7bc63", "#d8ba55", "#f8ba43", "#fbcd2d", "#f4e41c",
                "#f8fa0d"]
    cmap = colors.ListedColormap(pallette, name="from_list", N=None)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, px in enumerate(original_x):
        if original_y[i] == 0:
            pixels[i] = 0
        else:
            pixels[i] = clf.predict(px.reshape(1, -1))

    # pixels = clf.predict(original)
    pixels = pixels.reshape(args.image_height, args.image_width)
    pixels = np.uint8(mapper.to_rgba(pixels)*255)
    img = Image.fromarray(pixels)
    img.save(args.output_path)
    # img.show()

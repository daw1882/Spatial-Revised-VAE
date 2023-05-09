from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colormaps


if __name__ == '__main__':
    df = pd.read_csv("output.csv", header=None)
    print(df.head())
    X = df.iloc[:, :-2].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    print(np.unique(Y))
    print(X, X.shape)
    print(Y, Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.70, random_state=42
    )
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    exit()

    print("--------------------")
    print("Training...")
    # clf = svm.SVC()
    clf = KNeighborsClassifier(3)
    clf.fit(x_train, y_train)
    print("Training done!")
    print("--------------------")

    y_pred = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("--------------------")
    print("Making image...")
    print(X.shape)
    pixels = clf.predict(X)
    pixels = pixels.reshape(329, 599)
    print(pixels.shape)
    cmap = colormaps['Set3']
    pixels = np.uint8(cmap(pixels)*255)
    print(pixels, pixels.shape)
    img = Image.fromarray(pixels)
    img.save('knn.png')
    # img.show()

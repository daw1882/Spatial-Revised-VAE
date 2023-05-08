from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from PIL import Image


if __name__ == '__main__':
    df = pd.read_csv("output.csv", header=None)
    print(df.head())
    X = df.iloc[:, :-2].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    print(X, X.shape)
    print(Y, Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.70, random_state=42
    )
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    print("--------------------")
    print("Training...")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print("Training done!")
    print("--------------------")

    # y_pred = clf.predict(x_test)
    # print("Accuracy:", accuracy_score(y_test, y_pred))

    print("--------------------")
    print("Making image...")
    pixels = clf.predict(X)
    pixels = pixels.reshape((329, 599))

    img = Image.fromarray(pixels).convert('RGB')
    img.save('test.png')
    img.show()

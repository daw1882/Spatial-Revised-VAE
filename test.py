import scipy
import numpy as np


if __name__ == '__main__':
    mat = scipy.io.loadmat('C:/Users/dade_/NN_DATA/PaviaU.mat')
    print(mat.keys())
    print(mat["paviaU"].shape)
    print(mat["paviaU"])

    mat = scipy.io.loadmat('C:/Users/dade_/NN_DATA/PaviaU_gt.mat')
    print(mat.keys())
    print(mat["paviaU_gt"].shape)
    print(mat["paviaU_gt"])
    test = np.transpose(mat["paviaU_gt"])
    r, c = test.shape
    print(test[0:r-11, 0:c-11].shape)

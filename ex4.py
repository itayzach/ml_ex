import os
import sys
import ml_common as ml
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))
from sklearn.preprocessing import OneHotEncoder


########################################################################
# main
########################################################################
def main():
    print("====================================================")
    print("Neural network - two layers")
    print("Classify MNIST data set")
    print("====================================================")

    # Load data
    data = loadmat("data/ex3data1")

    # Convert to matrices
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])
    print("X is a 5000 samples of 20x20=400 pixels handwritten digits images")
    print("y is the classification of each image (1-9 are 1-9, 0 is 10)")
    print("X shape = " + str(X.shape))
    print("y shape = " + str(y.shape))

    num_samples, num_features = X.shape

    # Load given weights
    given_weights = loadmat("data/ex4weights")
    given_theta1 = np.matrix(given_weights['Theta1'])
    given_theta2 = np.matrix(given_weights['Theta2'])
    print("Theta1 shape = " + str(given_theta1.shape))
    print("Theta2 shape = " + str(given_theta2.shape))

    # Transform y to one-hot vectors. for example, 3 --> [ 0 0 1 0 0 0 0 0 0 0 ]
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)

    loss = ml.nnLoss(X, given_theta1, given_theta2, y_onehot)
    print loss

    llambda = 1.0
    loss_reg = ml.nnLossRegularized(X, given_theta1, given_theta2, y_onehot, llambda)
    print loss_reg


if __name__ == "__main__":
    main()

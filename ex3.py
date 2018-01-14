import os
import sys
import ml_common as ml
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))


########################################################################
# main
########################################################################
def main():
    print("====================================================")
    print("Multi class classification with Logistic regression")
    print("Classify MNIST data set")
    print("====================================================")

    # Load data
    data = loadmat("data/ex3data1")
    print("X is a 5000 samples of 20x20=400 pixels handwritten digits images")
    print("y is the classification of each image (1-9 are 1-9, 0 is 10)")
    print("X shape = " + str(data['X'].shape))
    print("y shape = " + str(data['y'].shape))

    # Convert to matrices
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])
    num_samples, num_features = X.shape

    # Add column of zeros to feature matrix (bias)
    X = np.insert(X, 0, values=np.ones(num_samples), axis=1)

    # Calculate weights matrix such that column k contains the linear regression weights
    # for the classification of k vs not k digit
    w_matrix = ml.oneVsAll(X, y, ml.sigmoid, llambda=1, num_labels=10)

    # Run predictions for each column and take the argmax with sigmoid to have the maximum probability
    y_pred = ml.predict_all(X, w_matrix)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = {0}%'.format(accuracy * 100)


if __name__ == "__main__":
    main()

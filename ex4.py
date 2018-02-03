import os
import sys
import ml_common as ml
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


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

    print("====================================================")
    print("Working with the given weights")
    print("====================================================")
    loss = ml.nnLoss(X, given_theta1, given_theta2, y_onehot)
    print "loss of given weights (no reg) = " + str(loss)

    llambda = 1.0
    loss_reg = ml.nnLossRegularized(X, given_theta1, given_theta2, y_onehot, llambda)
    print "loss of given weights (with reg. llambda = " + str(llambda) + ") = " + str(loss_reg)

    num_hidden = 25
    num_labels = 10
    print("====================================================")
    print("NN weights")
    print("====================================================")
    print("number of possible labels = " + str(num_labels))
    print("size of hidden layer = " + str(num_hidden))
    # Initiate theta1 and theta2 matrices (includes the bias)
    theta1 = ml.nnInitWeights(num_hidden, num_features+1)
    theta2 = ml.nnInitWeights(num_labels, num_hidden+1)

    # Since the minimize function cannot have two matricies as x0 (init params)
    # a single long vector is generated:
    theta12_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

    # Run!
    fmin = minimize(fun=ml.nnBackProp,            # objective function to minimize
                    x0=theta12_vec,          # initial set of parameters
                    args=(X, y_onehot, llambda),  # arguments to objective
                    method='TNC',                 # minimization method
                    jac=True,                     # jacobian
                    options={'maxiter': 250})     # limit to 250 iterations

    # Reshape to two weights matricies
    theta12_vec = fmin.x
    theta1_mat = np.matrix(np.reshape(theta12_vec[:num_hidden * (num_features + 1)], (num_hidden, num_features + 1)))
    theta2_mat = np.matrix(np.reshape(theta12_vec[num_hidden * (num_features + 1):], (num_labels, num_hidden + 1)))

    print("====================================================")
    print("Check accuracy of calculated weights")
    print("====================================================")
    # Perform forward pass with calculated weights
    a1, z2, a2, z3, h = ml.nnForwardPass(X, theta1_mat, theta2_mat)

    # Take the maximum of probabilty argument
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = {0}%'.format(accuracy * 100)

if __name__ == "__main__":
    main()

import os
import sys
import ml_common as ml
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))


########################################################################
# main
########################################################################
def main():
    print_plots_flag = False

    # Load data
    data = loadmat("../data/ex3data1")
    print("X is a 5000 samples of 20x20=400 pixels handwritten digits images")
    print("y is the classification of each image (1-9 are 1-9, 0 is 10)")
    print("X shape = " + str(data['X'].shape))
    print("y shape = " + str(data['y'].shape))

    X = data['X']
    y = data['y']
    num_samples, num_features = X.shape
    # Add column of zeros to feature matrix (bias)
    X = np.insert(X, 0, values=np.ones(num_samples), axis=1)

    w = np.matrix(np.zeros((num_features+1, 1)))
    # llambda = 0  # regularization factor
    # gradient = ml.grad(w, X, y, ml.sigmoid, llambda)
    # print("grad = " + str(gradient))
    #
    loss = ml.logRegLossRegularized(w, X, y, ml.sigmoid, llambda=1)

    w_matrix = ml.oneVsAll(X, y, ml.sigmoid, llambda=1, num_labels=10)
    print X.shape, w_matrix.shape
    y_pred = ml.predict_all(X, w_matrix)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = {0}%'.format(accuracy * 100)

    exit(0)

    # # Print stats
    # print("====================================================")
    # print("Head of data")
    # print("====================================================")
    # print(data.head())
    #
    # # Extract +/-1 outcomes for admissions
    # # Set X and y
    # cols = data.shape[1]  # shape returns tuple of (cols, rows)
    # X = data.iloc[:, 0:cols-1]
    # y = data.iloc[:, cols-1:cols]
    #
    # # Convert DataFrame to Matrix
    # X = np.matrix(X.values)
    # y = np.matrix(y.values)
    #
    # iters = 100000
    # w_gd, loss_vec = ml.gradientDescent(X, y, alpha=0.01, iters=iters, h=ml.sigmoid, loss=ml.logRegLoss)
    #
    # # print(loss_vec)
    # print(w_gd)
    #
    # num_samples, num_features = X.shape
    # init_w = np.matrix(np.zeros((num_features, 1)))
    # result = opt.fmin_tnc(func=ml.logRegLoss, x0=init_w, fprime=ml.grad, args=(X, y, ml.sigmoid), disp=False)
    # w_tcn = result[0]
    # print(w_tcn)
    #
    # if print_plots_flag:
    #     x_axis = np.linspace(min(min(positive['Exam 1']), min(negative['Exam 1'])), max(max(positive['Exam 1']), max(negative['Exam 1'])), 100)
    #     # w1x + w2y + w0 = 0
    #     # y = -(w1/w2)x - w0/w1
    #     f_gd = - (w_gd[1, 0]/w_gd[2, 0] * x_axis) -w_gd[0, 0]/w_gd[2, 0]
    #     f_tcn = - (w_tcn[1]/w_tcn[2] * x_axis) - w_tcn[0]/w_tcn[2]
    #
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.plot(x_axis, f_gd, 'r', label='Prediction with gradient descent. #iters = ' + str(iters))
    #     ax.plot(x_axis, f_tcn, 'g', label='Prediction with fmin function')
    #     ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    #     ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    #     ax.legend(loc=2)
    #     ax.set_xlabel('Exam 1 Score')
    #     ax.set_ylabel('Exam 2 Score')
    #     plt.show()


if __name__ == "__main__":
    main()

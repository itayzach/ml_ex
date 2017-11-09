import os
import sys
import ml_common as ml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))


########################################################################
# main
########################################################################
def main():
    print_plots_flag = False

    # Load data
    path = os.getcwd() + '/../data/ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

    positive = data[data['Admitted'] == 1]
    negative = data[data['Admitted'] == 0]
    #print(positive)

    # Add column of zeros to feature matrix (bias)
    data.insert(0, 'Bias', 1)

    # Extract +/-1 outcomes for admissions
    # Set X and y
    cols = data.shape[1]  # shape returns tuple of (cols, rows)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    if print_plots_flag:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
        ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
        ax.legend()
        ax.set_xlabel('Exam 1 Score')
        ax.set_ylabel('Exam 2 Score')

        plt.show()

    w_gd, loss_vec = ml.gradientDescent(X, y, alpha=0.01, iters=1000, h=ml.sigmoid, loss=ml.logRegLoss)

    # print(loss_vec)
    print(w_gd)

    num_samples, num_features = X.shape
    init_w = np.matrix(np.zeros((num_features, 1)))
    w_tcn = opt.fmin_tnc(func=ml.logRegLoss, x0=init_w, fprime=ml.grad, args=(X, y, ml.sigmoid), disp=False)
    print(w_tcn)
if __name__ == "__main__":
    main()

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
    print("====================================================")
    print("Logistic regression")
    print("Predict admission according to Exams 1 & 2 results")
    print("====================================================")
    print_plots_flag = True

    # Load data
    path = os.getcwd() + '/data/ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

    positive = data[data['Admitted'] == 1]
    negative = data[data['Admitted'] == 0]

    # Add column of zeros to feature matrix (bias)
    data.insert(0, 'Bias', 1)

    # Print stats
    print("====================================================")
    print("Head of data")
    print("====================================================")
    print(data.head())
    print("====================================================")

    # Extract +/-1 outcomes for admissions
    # Set X and y
    cols = data.shape[1]  # shape returns tuple of (cols, rows)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # run fmin_tcn to find best weights
    num_samples, num_features = X.shape
    init_w = np.zeros((num_features, 1))
    result = opt.fmin_tnc(func=ml.logRegLoss, x0=init_w, fprime=ml.grad, args=(X, y, ml.sigmoid), disp=False)
    w_tcn = result[0]
    print("Weights from TCN:")
    print(w_tcn)

    if print_plots_flag:
        x_axis = np.linspace(min(min(positive['Exam 1']), min(negative['Exam 1'])), max(max(positive['Exam 1']), max(negative['Exam 1'])), 100)
        # w1x + w2y + w0 = 0
        # y = -(w1/w2)x - w0/w1
        f_tcn = - (w_tcn[1]/w_tcn[2] * x_axis) - w_tcn[0]/w_tcn[2]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x_axis, f_tcn, 'g', label='Prediction with fmin function')
        ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
        ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
        ax.legend(loc=2)
        ax.set_xlabel('Exam 1 Score')
        ax.set_ylabel('Exam 2 Score')
        plt.show()


if __name__ == "__main__":
    main()

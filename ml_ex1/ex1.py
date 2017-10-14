import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import my_lin_reg_lib as lr

########################################################################
# main
########################################################################
def main():
    # Load data
    path = os.getcwd() + '/../data/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # Plot data
    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    # plt.show()

    # Add column of zeros to feature matrix (bias)
    data.insert(0, 'Bias', 1)

    # Print stats
    print("====================================================")
    print("Head of data")
    print("====================================================")
    print(data.head())

    # Set X and y
    cols = data.shape[1]  # shape returns tuple of (cols, rows)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # Call gradientDescent with learning rate of alpha
    alpha = 0.01
    iters = 1000
    w, cost_vec = lr.gradientDescent(X, y, alpha, iters)

    # Plot data and prediction
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = w[0, 0] + (w[1, 0] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')

    # Plot error vs. training epoch (iteration)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost_vec, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


if __name__ == "__main__":
    main()

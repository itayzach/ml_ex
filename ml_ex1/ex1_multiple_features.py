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
    path = os.getcwd() + '/../data/ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

    # Print stats
    print("====================================================")
    print("Head of data (before normalization)")
    print("====================================================")
    print(data.head())

    # Normalize data
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    print("====================================================")
    print("Data Mean")
    print("====================================================")
    print(data_mean)
    print("====================================================")
    print("Data STD")
    print("====================================================")
    print(data_std)
    # Print stats
    print("====================================================")
    print("Head of data (after normalization)")
    print("====================================================")
    print(data.head())
    # Add column of zeros to feature matrix (bias)
    data.insert(0, 'Bias', 1)

    # Set X and y
    cols = data.shape[1]  # shape returns tuple of (rows, cols)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # Call gradientDescent with learning rate of alpha
    alpha = 0.01
    iters = 1000
    w, cost = lr.gradientDescent(X, y, alpha, iters)

    print("====================================================")
    print("Final weights = \n" + str(w))
    print("====================================================")
    # Plot data and prediction
    x = np.linspace(data.Size.min(), data.Size.max(), 100)
    f = w[0, 0] + (w[1, 0] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Size, data.Price, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Size')
    ax.set_ylabel('Price')
    ax.set_title('Predicted Price vs. House Size (square feet)')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Bedrooms, data.Price, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Bedrooms')
    ax.set_ylabel('Price')
    ax.set_title('Predicted Price vs. House Bedrooms')

    # Plot error vs. training epoch (iteration)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


if __name__ == "__main__":
    main()

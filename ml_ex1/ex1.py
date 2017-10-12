import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

########################################################################
# gradientDescent
########################################################################
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

########################################################################
# computeCost
########################################################################
def computeCost(X, y, w):
    inner = np.power(((X * w.T) - y), 2)  # (<X, w> - y)^2
    cost = np.sum(inner) / (2 * len(X))
    return cost

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
    cols = data.shape[1]  # shape returns tuple of (97,3)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    w = np.matrix(np.array([0, 0]))

    # Call gradientDescent with learning rate of alpha
    alpha = 0.01
    iters = 1000
    g, cost = gradientDescent(X, y, w, alpha, iters)

    # Plot data and prediction
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    # Plot error vs. training epoch (iteration)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


if __name__ == "__main__":
    main()

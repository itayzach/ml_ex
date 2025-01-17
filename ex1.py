import os
import sys
import ml_common as ml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_common')))


########################################################################
# main
########################################################################
def main():
    print("====================================================")
    print("Linear regression")
    print("Predict profit according to population size")
    print("====================================================")
    print_plots_flag = True
    # Load data
    path = os.getcwd() + '/data/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

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
    iters = 100000
    w_gd, loss_vec = ml.gradientDescent(X, y, alpha=alpha, iters=iters, h=ml.innerProd, loss=ml.linRegLoss)

    # Calculate weights using normal equation
    w_pseudo_inv = (np.linalg.inv(X.T*X))*X.T*y

    # Print final weights
    print("====================================================")
    print("Gradient descent")
    print("----------------------------------------------------")
    print("Iterations = " + str(iters))
    print("Alpha      = " + str(alpha))
    print("Weights    = " + str(w_gd[0, 0]) + ", " + str(w_gd[1, 0]))
    print("----------------------------------------------------")
    print("Pseudo inverse")
    print("----------------------------------------------------")
    print("Weights    = " + str(w_pseudo_inv[0, 0]) + ", " + str(w_pseudo_inv[1, 0]))

    # Check for a specific prediction both methods
    new_data = pd.DataFrame({'Population': [5.]})
    new_data.insert(0, 'Bias', 1.)
    new_X = np.matrix(new_data.values)
    predicted_price_gd = new_X*w_gd
    predicted_price_pseudo_inv = new_X*w_pseudo_inv
    print("====================================================")
    print("Prediction for : \n" + str(new_data) + "\n")
    print("Predicted profit with gradient descent = ")
    print("\t\t" + str(predicted_price_gd[0, 0]))
    print("Predicted profit with pseudo inverse   = ")
    print("\t\t" + str(predicted_price_pseudo_inv[0, 0]))
    print("====================================================")
# Plot data
    if print_plots_flag:
        x = np.linspace(data.Population.min(), data.Population.max(), 100)
        f = w_gd[0, 0] + (w_gd[1, 0] * x)
        f2 = w_pseudo_inv[0, 0] + (w_pseudo_inv[1, 0] * x)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x, f, 'r', label='Prediction with gradient descent. #iters = ' + str(iters))
        ax.plot(x, f2, 'g', label='Prediction with pseudo inverse')
        ax.scatter(data.Population, data.Profit, label='Training Data')
        ax.legend(loc=2)
        ax.set_xlabel('Population')
        ax.set_ylabel('Profit')
        ax.set_title('Predicted Profit vs. Population Size')

        # Plot error vs. training epoch (iteration)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(iters), loss_vec, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()

if __name__ == "__main__":
    main()

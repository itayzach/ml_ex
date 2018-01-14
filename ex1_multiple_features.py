import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import my_lin_reg_lib as lr


########################################################################
# main
########################################################################
def main():
    print_plots_flag = True
    # Load data
    path = os.getcwd() + '/../data/ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

    # Set X and y
    cols = data.shape[1]  # shape returns tuple of (rows, cols)
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # Normalize features
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std

    # Add column of zeros to feature matrix (bias)
    X.insert(0, 'Bias', 1.)

    # Print stats
    print("====================================================")
    print("Data before and after normalization")
    print("====================================================")
    print(pd.concat([data.head(), X.head(), y.head()], axis=1))

    # Convert DataFrame to Matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # Call gradientDescent with learning rate of alpha
    alpha = 0.01
    iters = 10000
    w_gd, cost = lr.gradientDescent(X, y, alpha, iters)

    # Plot data and prediction
    if print_plots_flag:
        # Plot error vs. training epoch (iteration)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(iters), cost, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()

    # Calculate weights using normal equation
    w_pseudo_inv = (np.linalg.inv(X.T*X))*X.T*y

    # Print final weights
    print("====================================================")
    print("Final weights gradient descent = \n" + str(w_gd))
    print("Final weights pseudo inverse   = \n" + str(w_pseudo_inv))
    print("====================================================")

    # Check for a specific prediction both methods
    new_data = pd.DataFrame({'Size': [1650.], 'Bedrooms': [3.]})
    new_data_normalized = (new_data - X_mean) / X_std
    new_data_normalized.insert(0, 'Bias', 1.)
    new_X = np.matrix(new_data_normalized.values)
    predicted_price_gd = new_X*w_gd
    predicted_price_pseudo_inv = new_X*w_pseudo_inv
    print("Prediction for : \n" + str(new_data) + "\n")
    print("Predicted price with gradient descent = " + str(predicted_price_gd[0, 0]))
    print("Predicted price with pseudo inverse   = " + str(predicted_price_pseudo_inv[0, 0]))
    print("====================================================")

if __name__ == "__main__":
    main()

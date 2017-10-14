import numpy as np


########################################################################
# gradientDescent
########################################################################
def gradientDescent(X, y, alpha, iters):
    # Initialize
    num_samples, num_features = X.shape
    w = np.matrix(np.zeros((num_features, 1)))
    cost_vec = np.zeros(iters)

    for i in range(iters):
        cost_vec[i] = computeCost(X, y, w)
        # Calculate gradient
        grad = (0.5/num_samples) * X.T * (X*w - y)
        # Update weights
        w = w - alpha*grad

    return w, cost_vec


########################################################################
# computeCost
########################################################################
def computeCost(X, y, w):
    norm = np.linalg.norm(X*w - y)  # ||Xw - y||
    norm_squared = np.power(norm, 2)  # ||Xw - y||^2
    cost = norm_squared / (2 * len(X))
    return cost



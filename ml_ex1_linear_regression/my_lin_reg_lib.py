import numpy as np


########################################################################
# innerProd
########################################################################
def innerProd(X, w):
    return X*w


########################################################################
# gradientDescent
########################################################################
def gradientDescent(X, y, alpha, iters, h):
    # Initialize
    num_samples, num_features = X.shape
    w = np.matrix(np.zeros((num_features, 1)))
    cost_vec = np.zeros(iters)

    for i in range(iters):
        cost_vec[i] = computeCost(X, y, w, h)
        # Calculate gradient
        grad = (0.5/num_samples) * X.T * (h(X, w) - y)
        # Update weights
        w = w - alpha*grad

    return w, cost_vec


########################################################################
# computeCost
########################################################################
def computeCost(X, y, w, h):
    norm = np.linalg.norm(h(X, w) - y)  # ||h(X,w) - y||
    norm_squared = np.power(norm, 2)  # ||h(X,w) - y||^2
    cost = norm_squared / (2 * len(X))
    return cost



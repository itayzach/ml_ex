import numpy as np


########################################################################
# sigmoid
########################################################################
def sigmoid(X, w):
    w = w.reshape(w.size, 1)
    z = X*w
    return 1/(1 + np.exp(-z))


########################################################################
# innerProd
########################################################################
def innerProd(X, w):
    return X*w


########################################################################
# grad
########################################################################
def grad(w, X, y, h):
    num_samples, num_features = X.shape
    return (0.5/num_samples) * X.T * (h(X, w) - y)


########################################################################
# gradientDescent
########################################################################
def gradientDescent(X, y, alpha, iters, h, loss):
    # Initialize
    num_samples, num_features = X.shape
    w = np.matrix(np.zeros((num_features, 1)))
    loss_vec = np.zeros(iters)

    for i in range(iters):
        loss_vec[i] = loss(w, X, y, h)
        # Calculate gradient
        g = grad(w, X, y, h)
        # Update weights
        w = w - alpha*g

    return w, loss_vec


########################################################################
# linRegLoss
########################################################################
def linRegLoss(w, X, y, h):
    norm = np.linalg.norm(h(X, w) - y)  # ||h(X,w) - y||
    norm_squared = np.power(norm, 2)  # ||h(X,w) - y||^2
    loss = norm_squared / (2 * len(X))
    return loss


########################################################################
# logRegLoss
########################################################################
def logRegLoss(w, X, y, h):
    logs_diff = -y.T*np.log(h(X, w)) - (1-y).T*np.log(1-h(X, w))
    loss = logs_diff / (2 * len(X))
    return loss

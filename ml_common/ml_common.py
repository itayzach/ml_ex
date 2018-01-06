import numpy as np
from scipy.optimize import minimize


########################################################################
# handleMatDir
########################################################################
def handleMatDir(w, X, y):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)

    if w.shape[1] != 1:
        w = w.T
    if y.shape[1] != 1:
        y = y.T
    return w, X, y


########################################################################
# sigmoid
########################################################################
# def sigmoid(X, w):
#     #if X.shape[1] != w.shape[0]:
#     w = w.reshape(w.size, 1)
#     z = np.dot(X, w)
#     return 1/(1 + np.exp(-z))

def sigmoid(z):
    return 1/(1 + np.exp(-z))


########################################################################
# innerProd
########################################################################
def innerProd(z):
    return z
    # return X*w


########################################################################
# grad
########################################################################
def grad(w, X, y, h):
    w, X, y = handleMatDir(w, X, y)
    num_samples = X.shape[0]

    g = (1./num_samples) * X.T * (h(X * w) - y)
    return g


########################################################################
# grad with regularization
########################################################################
def gradRegularized(w, X, y, h, llambda):
    w, X, y = handleMatDir(w, X, y)

    error = h(X * w) - y
    g = ((X.T * error) / len(X)) + ((llambda / len(X)) * w)

    # intercept gradient is not regularized
    g[0, 0] = (error.T * X[:, 0]) / len(X)
    return g


########################################################################
# gradientDescent
########################################################################
def gradientDescent(X, y, alpha, iters, h, loss):
    X = np.matrix(X)
    y = np.matrix(y)
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
    w, X, y = handleMatDir(w, X, y)
    norm = np.linalg.norm(h(X * w) - y)  # ||h(X,w) - y||
    norm_squared = np.power(norm, 2)    # ||h(X,w) - y||^2
    loss = norm_squared / (2 * len(X))
    return loss


########################################################################
# logRegLoss
########################################################################
def logRegLoss(w, X, y, h):
    w, X, y = handleMatDir(w, X, y)
    logs_diff = -y.T*np.log(h(X * w)) - (1-y).T*np.log(1-h(X * w))
    loss = logs_diff / (1 * len(X))
    return loss


########################################################################
# logRegLoss regularized
########################################################################
def logRegLossRegularized(w, X, y, h, llambda):
    w, X, y = handleMatDir(w, X, y)
    w_norm_squared = np.power(np.linalg.norm(w[1:]), 2)  # ||w[1:]||^2
    loss = logRegLoss(w, X, y, h) + (llambda / (2 * len(X))) * w_norm_squared
    return loss


########################################################################
# oneVsAll
########################################################################
def oneVsAll(X, y, h, llambda, num_labels):
    num_samples, num_features = X.shape
    # each row k of w will contain the weights for the k'th label. (here, "label" refers to a digit 1-9,0)
    w_matrix = np.matrix(np.zeros((num_labels, num_features)))

    # loop over all labels (1-9 and 10 that represents "0")
    for k in range(1, num_labels + 1):
        w_k = np.zeros((num_features, 1))
        # y_k has "1" where y is k, and "0" for all other labels
        y_k = np.matrix([1 if label == k else 0 for label in y])
        y_k = np.reshape(y_k, (num_samples, 1))
        fmin = minimize(fun=logRegLossRegularized,       # objective function to minimize
                        x0=w_k,                          # initial set of parameters
                        args=(X, y_k, h, llambda),       # arguments to objective
                        method='TNC',                    # minimization method
                        jac=gradRegularized)             # jacobian
        w_matrix[k-1, :] = fmin.x

    return w_matrix


########################################################################
# predict_all
########################################################################
def predict_all(X, w_matrix):
    num_samples, num_features = X.shape
    num_labels = w_matrix.shape[0]

    # convert to matrices
    X = np.matrix(X)
    w_matrix = np.matrix(w_matrix).reshape(num_labels, num_features)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * w_matrix.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


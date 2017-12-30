import numpy as np
from scipy.optimize import minimize


########################################################################
# sigmoid
########################################################################
def sigmoid(X, w):
    #if X.shape[1] != w.shape[0]:
    w = w.reshape(w.size, 1)
    z = np.dot(X, w)
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
    gradient = (0.5/num_samples) * X.T * (h(X, w) - y)
    return gradient


########################################################################
# grad with regularization
########################################################################
def gradRegularized(w, X, y, h, llambda):
    num_samples, num_features = X.shape
    reg = np.matrix(llambda/num_samples * w).reshape(num_features, 1)
    gradient = grad(w, X, y, h) + reg
    # the bias term should not have regularization
    gradient[0, 0] = (0.5/num_samples) * X[:, 0].T * (h(X, w) - y)
    return gradient


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


########################################################################
# logRegLoss regularized
########################################################################
def logRegLossRegularized(w, X, y, h, llambda):
    # note that w0 is not regularized
    w_norm_squared = np.power(np.linalg.norm(w[1:]), 2)  # ||w[1:]||^2
    loss = logRegLoss(w, X, y, h) + (llambda/(2*len(X)))*w_norm_squared
    return loss


########################################################################
# oneVsAll
########################################################################
def oneVsAll(X, y, h, llambda, num_labels):
    num_samples, num_features = X.shape
    # each row k of w will contain the weights for the k'th label. (here, "label" refers to a digit 1-9,0)
    w_matrix = np.zeros((num_labels, num_features))

    # loop over all labels (1-9 and 10 that represents "0")
    for k in range(1, num_labels + 1):
        # w_k = np.zeros(num_features)
        w_k = np.matrix(np.zeros((num_features, 1)))
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


def predict_all(X, w_matrix):
    num_samples, num_features = X.shape
    num_labels = w_matrix.shape[0]

    print X.shape, w_matrix.shape
    # convert to matrices
    X = np.matrix(X)
    w_matrix = np.matrix(w_matrix).reshape(num_labels, num_features)

    # compute the class probability for each class on each training instance
    h = np.matrix(np.zeros((num_samples, num_labels)))
    print X.shape, h.shape

    # TODO: sigmoid has a bug. it cannot get a matrix, only a vector
    for k in range(1, num_labels):
        predict = sigmoid(X, w_matrix[k-1, :])
        h[:, k-1] = predict

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax



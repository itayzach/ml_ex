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


def sigmoidGrad(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z))) # element-wise

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


########################################################################
# nnForwardPass
########################################################################
def nnForwardPass(X, theta1_mat, theta2_mat):
    num_samples, num_features = X.shape

    a1 = X
    a1 = np.insert(a1, 0, values=np.ones(num_samples), axis=1)  # add bias
    z2 = a1*theta1_mat.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(num_samples), axis=1)  # add bias
    z3 = a2*theta2_mat.T
    y_hat = sigmoid(z3)

    return a1, z2, a2, z3, y_hat


########################################################################
# nnLoss
########################################################################
def nnLoss(X, theta1_mat, theta2_mat, y):
    num_labels = y.shape[0] # y is a one-hot vector, therefor its length is the number of possible labels
    loss_sum = 0

    # predict
    a1, z2, a2, z3, y_nn = nnForwardPass(X, theta1_mat, theta2_mat)

    # convert to matrix type
    y_nn = np.matrix(y_nn)
    y = np.matrix(y)

    for k in range(num_labels):
        y_k = y[k, :]
        y_nn_k = y_nn[k, :]

        logs_diff = -y_k * np.log(y_nn_k).T - (1 - y_k) * np.log(1 - y_nn_k).T
        loss_sum += logs_diff

    loss = loss_sum / len(X)

    return loss


########################################################################
# nnLossRegularized
########################################################################
def nnLossRegularized(X, theta1_mat, theta2_mat, y, llambda):

    theta1_mat_norm_squared = np.power(np.linalg.norm(theta1_mat), 2)
    theta2_mat_norm_squared = np.power(np.linalg.norm(theta2_mat), 2)

    regularization = (float(llambda) / (2 * len(X))) * (theta1_mat_norm_squared + theta2_mat_norm_squared)
    loss = nnLoss(X, theta1_mat, theta2_mat, y) + regularization

    return loss


########################################################################
# nnInitWeights
########################################################################
def nnInitWeights(num_rows, num_cols):
    eps = 0.12
    theta_mat = np.random.uniform(low=-eps, high=eps, size=(num_rows, num_cols))

    return theta_mat


########################################################################
# nnBackProp
########################################################################
def nnBackProp(theta12_vec, X, y, llambda):
    num_samples, num_features = X.shape
    num_hidden = 25
    num_labels = y.shape[1]

    # reshape to matrices
    theta1_mat = np.matrix(np.reshape(theta12_vec[:num_hidden * (num_features + 1)], (num_hidden, num_features + 1)))
    theta2_mat = np.matrix(np.reshape(theta12_vec[num_hidden * (num_features + 1):], (num_labels, num_hidden + 1)))

    # forward pass
    a1, z2, a2, z3, y_hat = nnForwardPass(X, theta1_mat, theta2_mat)

    # calculate the loss
    loss = nnLossRegularized(X, theta1_mat, theta2_mat, y, llambda)

    delta1 = np.matrix(np.zeros(theta1_mat.shape))
    delta2 = np.matrix(np.zeros(theta2_mat.shape))

    # backward pass
    for t in range(num_samples):
        a1_t = a1[t, :]  # (1, 401)
        z2_t = z2[t, :]  # (1, 25)
        a2_t = a2[t, :]  # (1, 26)
        y_hat_t = y_hat[t, :]  # (1, 10)
        y_t = y[t, :]  # (1, 10)

        # step 2
        d3_t = y_hat_t - y_t # (1, 10)

        # step 3
        z2_t = np.insert(z2_t, 0, values=np.ones(1), axis=1)  # (1, 26)
        d2_t = np.multiply(d3_t*theta2_mat, sigmoidGrad(z2_t))  # (1,10)*(10,26) .* (1,26) = (1,26)

        # step 4
        delta1 = delta1 + d2_t[:, 1:].T*a1_t  # (25,401) + (25,1)*(1,401)
        delta2 = delta2 + d3_t.T*a2_t  # (10,1)*(1,26)

    delta1 /= len(X)
    delta2 /= len(X)

    # regularize
    delta1[:, 1:] = delta1[:, 1:] + (theta1_mat[:, 1:] * llambda) / len(X)
    delta2[:, 1:] = delta2[:, 1:] + (theta2_mat[:, 1:] * llambda) / len(X)

    # concatenate to a single long vector of weights
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return loss, grad



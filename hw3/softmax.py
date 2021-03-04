import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    #############################################################################
    panel, soft = 0, 0 
    K = theta.shape[1]
    # calculate the penalty term
    for j in range(dim):
        for k in range(K):
            panel += theta[j][k]**2
    panel *= reg/2/m
    
    # calculate soft term
    for i in range(m):
        for k in range(K):
            if y[i] == k:
                top = np.exp((theta[:,k].T).dot(X[i]))
                vec = np.exp(np.matmul(X[i],theta)-max(np.matmul(X[i],theta)))
                S = np.exp(np.matmul(X[i],theta)-max(np.matmul(X[i],theta))).sum(axis = 0)
                bot = (vec/S).sum(axis = 0)
                    
       

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

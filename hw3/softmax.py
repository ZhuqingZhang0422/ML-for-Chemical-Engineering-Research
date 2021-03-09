import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
    #Softmax loss function, naive implementation (with loops)
    #Inputs:
    #- theta: d x K parameter matrix. Each column is a coefficient vector for class k
    #- X: m x d array of data. Data are d-dimensional rows.
    #- y: 1-dimensional array of length m with labels 0...K-1, for K classes
    #- reg: (float) regularization strength
    #Returns:
    #a tuple of:
    #- loss as single float
    #- gradient with respect to parameter matrix theta, an array of same size as theta
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
    
    # calculate soft term
    p_top = np.dot(X,theta)
    # conduct transformation to avoid overflow
    p_top = np.exp(p_top - np.max(p_top, axis = 1).reshape((-1,1)))
    p = p_top/np.sum(p_top, axis=1).reshape((-1,1))
    for i in range(m):
        for k in range(K):
            if y[i] == k:
                soft += np.log(p[i,k])
    
    # calculate the penalty term
    for j in range(dim):
        for k in range(K):
            panel += theta[j,k]*theta[j,k]
    panel *= reg/2/m
    J = -soft/m + panel
    
    # calculate the gradiant of loss function
    for k in range(K):
        for l in range(m):
            grad[:,k] += X[l,:]*((1 if y[l] == k else 0) - p[l,k])
        grad[:,k] /= (-m)
        grad[:,k] += reg/m*theta[:,k]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
 
  #Softmax loss function, vectorized version.
  #Inputs and outputs are the same as softmax_loss_naive.
  #Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################
    K = theta.shape[1]
    # compute probability matrixs
    p_top = np.dot(X,theta)
    # conduct transformation to avoid overflow
    p_top = np.exp(p_top - np.max(p_top, axis = 1).reshape((-1,1)))
    p = p_top/np.sum(p_top, axis=1).reshape((-1,1))
    soft, panel = 0, 0
    
    # calculate the panel term
    panel = reg/2/m*np.square(theta).sum(axis = 1).sum(axis = 0)
    
    # calculate the soft term
    m = int(X.shape[0])
    I = np.zeros((m,K))
    I[range(0,m), y.astype(int)] = 1
    soft = -(np.multiply(I,np.log(p))).sum(axis = 1).sum(axis = 0)/m
    J = panel + soft
    grad = -np.matmul(X.T,(I-p))/m + reg*theta/m 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
 #"""
 #SVM hinge loss function for two class problem

 #Inputs:
 #- theta: A numpy vector of size d containing coefficients.
 #- X: A numpy array of shape mxd 
 #- y: A numpy array of shape (m,) containing training labels; +1, -1
 #- C: (float) penalty factor

 #Returns a tuple of:
 #- loss as single float
 #- gradient with respect to theta; an array of same shape as theta
 #"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0 
  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  h = 1 - y*np.matmul(X,theta)
  J = np.square(theta).sum()/m/2 + np.maximum(h,np.zeros(h.shape)).sum()*C/m
  ind = (h > 0).astype(float)
  grad = theta/m + C/m*np.matmul(X.T,-y*ind)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, C):
  #"""
  #Structured SVM loss function, naive implementation (with loops).

  #Inputs have dimension d, there are K classes, and we operate on minibatches
  #of m examples.

  #Inputs:
  #- theta: A numpy array of shape d X K containing parameters.
  #- X: A numpy array of shape m X d containing a minibatch of data.
  #- y: A numpy array of shape (m,) containing training labels; y[i] = k means
  #  that X[i] has label k, where 0 <= k < K.
  #- C: (float) penalty factor

  #Returns a tuple of:
  #- loss J as single float
  #- gradient with respect to weights theta; an array of same shape as theta
  #"""

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  for i in range(m):
    # calculate the reference
    cor = np.dot(theta[:,y[i]],X[i,:])
    for j in range(K):
        if j != y[i]:
            diff = delta + np.dot(theta[:,j],X[i,:]) - cor
            if diff > 0:
                J += diff
                dtheta[:,j] += X[i,:]
                dtheta[:,y[i]] -= X[i,:]
  J = J/m*C + np.square(theta).sum(axis = 1).sum(axis = 0)/2/m
  dtheta = theta/m + C*dtheta/m
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may neßed to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return J, dtheta


def svm_loss_vectorized(theta, X, y, C):
  #"""
  #Structured SVM loss function, vectorized implementation.

  #Inputs and outputs are the same as svm_loss_naive.
  #"""
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0
  m = X.shape[0]
  K = theta.shape[1]
  d = theta.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################
  K, m = theta.shape[1], X.shape[0]
  # calculate the hinge term
  h  = np.matmul(X, theta)
  hh = h - h[range(len(y)),y].reshape((-1,1))
  hh[hh!=0] += delta
  l = np.maximum(0, hh)
  # add panely term
  J = np.sum(np.square(theta))/2/m + np.sum(l)/m*C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pick up the term which are predicted wrong
  wr_pred = (hh >0).astype(int)
  wr_pred[range(len(y)),y] = -(np.sum(wr_pred, axis=1))  
  dtheta = theta/m + np.matmul(X.T, wr_pred)/m*C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J , dtheta

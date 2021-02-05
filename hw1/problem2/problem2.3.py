import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def predict(T, query,Y, tau,iterations=100,lr=0.01):
  m = T.shape[0]
  a = np.ones((M, 1))
  X = np.hstack((T, a))
  qx = np.mat([query, 1])
  w = np.mat(np.eye(m))
  for i in range(m):
      xi = X[i]
      w[i, i] = np.exp(-np.dot((xi - qx), (xi - qx).T) / (2 * tau * tau))
  # calculating parameter theta
  theta=np.random.randn(X.shape[0],1)
  cost_history=np.zeros(iterations)
  theta_history = np.zeros(iterations)
  for it in range(iterations):
      prediction=np.dot(qx,theta)
      theta=theta-(1/m)*lr*np.matmul(w,(qx.T.dot((prediction-y))))
      theta_history[it,:]=theta.T
      cost=(m/2)*np.sum(np.square(prediction-y))
      cost_history[it] = cost
  return theta, theta_history, cost_history
import numpy as np
X = np.array([[1,0,3],[1,1,3],[1,0,1],[1,1,1]])
theta = np.array([0,-2,1])
y = np.array([1,1,0,0]) 

def sigmoid (z):
    #sig = np.zeros(z.shape)
    # Your code here
    z = np.array(z)
    sig = 1/ (1+ np.exp(-z))
    # End your code
    return sig

def grad_loss(theta,X,y,reg):
    m,dim = X.shape
    grad = np.zeros((dim,))
    penal_arr = np.hstack([[0],theta[1:]])
    grad = 1/m * np.matmul(X.T,sigmoid(np.matmul(X,theta))-y) + reg/m*penal_arr
    return grad
    
def Hessian (theta,X,reg):
    m,dim = X.shape
    diag_num = sigmoid(np.matmul(X,theta))*(1-sigmoid(np.matmul(X,theta)))
    S = np.diag(list(diag_num.reshape(-1)))
    penal = reg * np.diag([1]*theta.shape[0])
    H = 1/m * (np.matmul(np.matmul(X.T,S),X) + penal)
    return H

if __name__ == "__main__":
    for i in range (1,20):
        g = grad_loss(theta,X,y,reg = 0.07)
        H = Hessian(theta,X,reg = 0.07)
        theta_new = theta - np.matmul(np.linalg.inv(H),g)
        theta = theta_new
        print ("current_iteration: %d" %i, "theta = ", theta)
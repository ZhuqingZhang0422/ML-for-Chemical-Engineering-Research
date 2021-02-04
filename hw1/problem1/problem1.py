import numpy as np
import matplotlib.pyplot as plt
def categorical(x,p,num):
    a=np.random.uniform(size=100000)
    b=np.random.choice(a,num)
    c=[]
    d=[]
    for k in range(len(p)):
        d.append(sum(p[0:k+1]))
    for j in range(len(b)):
        for i in range(len(d)):
            if i == 0:
                if b[j] <= d[i]:
                    c.append(x[i])
            elif d[i - 1] < b[j] <= d[i]:
                c.append(x[i])
    return c
def unigau(mu,sigma,num):
    a = np.random.uniform(size=100000)
    b = np.random.choice(a, num)
    c = np.random.choice(a, num)
    z=np.multiply(np.sqrt(-2*np.log(b)),np.cos(2*np.pi*c))
    z=z*sigma+mu
    return z
def multigau(mu,cov,num,d):
    m=np.array(mu).reshape(d, 1)
    l=np.linalg.cholesky(cov)
    a = unigau(0,1,num*d).reshape(d,num)
    z=m+np.dot(l,a)
    return z
def mix(num,w,cov,mu,d):
    w=np.array(w)
    num_dis=len(w)
    z1 = np.zeros((num,num_dis))
    z2 = np.zeros((num, num_dis))
    for j in range(4):
        a=multigau(mu[j],cov[j],num,d[j])
        z1[:,j] = a[0]
        z2[:,j] = a[1]
    i=categorical(np.arange(num_dis),w,num)
    z1 = z1[np.arange(num), i]
    z2 = z2[np.arange(num), i]
    z = np.vstack((z1, z2))
    return z
####test 
cov=np.array([[1.,0.],[0.,1.]])
cov=[cov,cov,cov,cov]
mu=[[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]
mix(10,[0.25,0.25,0.25,0.25],cov,mu,[2,2,2,2])
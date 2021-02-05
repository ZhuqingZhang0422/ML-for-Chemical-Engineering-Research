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
####test1
a=categorical([1,2,3,4,5],[0.1,0.1,0.3,0.3,0.2],1000)
plt.hist(a,bins=5)
plt.xlabel('x', fontsize=20)
plt.ylabel('n', fontsize=20)
####test2
b=unigau(0,1,1000)
plt.hist(b,bins=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('n', fontsize=20)
###test3
c=multigau([1,1],[[1,0.5],[0.5,1]],1000,2)
plt.scatter(c[0],c[1])
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
####test4
cov=np.array([[1.,0.],[0.,1.]])
cov=[cov,cov,cov,cov]
mu=[[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]
d=mix(1000,[0.25,0.25,0.25,0.25],cov,mu,[2,2,2,2])
plt.scatter(d[0],d[1])
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
p=sum((np.power((d[0] - 0.1), 2) * np.power((d[1] - 0.2), 2))<=1)/1000
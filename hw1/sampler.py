import numpy as np
class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma,num):
        self.num = num
        self.mu = mu
        self.sigma = sigma
        pass

    def sample(self):
        a = np.random.uniform(size=100000)
        b = np.random.choice(a, self.num)
        c = np.random.choice(a, self.num)
        z = np.multiply(np.sqrt(-2 * np.log(b)), np.cos(2 * np.pi * c))
        z = z * self.sigma + self.mu
        return z
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.mu = Mu
        self.d = len(Mu)
        self.sigma = Sigma
        self.num = num

    def sample(self):
        m = np.array(self.mu).reshape(self.d, 1)
        l = np.linalg.cholesky(self.sigma)
        a = UnivariateNormal(0, 1, self.num * self.d).sample().reshape(self.d, self.num)
        z = m + np.dot(l, a)
        return z
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,x,ap,num):
        self.p = ap
        self.x = x
        self.num = num

    def sample(self):
        a = np.random.uniform(size=100000)
        b = np.random.choice(a, self.num)
        c = []
        d = []
        for k in range(len(self.p)):
            d.append(sum(self.p[0:k + 1]))
        for j in range(len(b)):
            for i in range(len(d)):
                if i == 0:
                    if b[j] <= d[i]:
                        c.append(self.x[i])
                elif d[i - 1] < b[j] <= d[i]:
                    c.append(self.x[i])
        return c


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,num,w,cov,mu,d):
        self.num = num
        self.w = w
        self.cov = cov
        self.mu = mu
        self.d = d

    def sample(self):
        w = np.array(self.w)
        num_dis = len(self.w)
        z1 = np.zeros((self.num, num_dis))
        z2 = np.zeros((self.num, num_dis))
        for j in range(4):
            a = MultiVariateNormal(self.mu[j], self.cov[j]).sample()
            z1[:, j] = a[0]
            z2[:, j] = a[1]
        i = Categorical(np.arange(num_dis), self.w, self.num).sample()
        z1 = z1[np.arange(self.num), i]
        z2 = z2[np.arange(self.num), i]
        z = np.vstack((z1, z2))
        return z

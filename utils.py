import numpy as np
import scipy.linalg as la
import scipy.stats as st



def sigmoid(t):
    return 1/(1 + np.exp(-t))

def gaussian_cdf(t):
    return st.norm.cdf(t)

def laplacian_cdf(t):
    sgn = np.sign(t)
    return (sgn + 1) / 2 - sgn * np.exp(-np.abs(t)) / 2 
    
def kendall_tau(x, y):
    # use kendalltau(x,y).statistic for SciPy version >=1.10
    return (1-st.kendalltau(x, y).correlation) / 2


model_dict = {"Bradley-Terry": sigmoid, "Thurstone": gaussian_cdf, "Laplacian CDF": laplacian_cdf}


class LST:
    def __init__(self, theta=None, link=sigmoid, n=5):
        if theta is None:
            self.n = n
            w = 2 * np.random.rand(self.n) - 1
            self.theta = np.sort(w, axis=None)[::-1]
        else:
            self.n = len(theta)
            self.theta = np.sort(theta, axis=None)[::-1]
        self.link = link
        self.tmat = np.zeros([self.n, self.n])
        self.nconst = self.n
        self.fmat = self.link(- self.theta.reshape(-1, 1) + self.theta.reshape(1, -1))
        self.tmat = self.fmat / self.nconst
        for i in range(self.n):
            rsum = np.sum(self.tmat[i, :]) - self.tmat[i, i]
            self.tmat[i, i] = 1 - rsum

    def sample(self, L=1):
        self.L = L
        self.yobs = np.zeros_like(self.tmat)
        for i in range(self.L):
            self.yobs += st.bernoulli.rvs(np.triu(self.fmat, 1)) / self.L
        self.yobs = np.triu(self.yobs, 1)
        self.yobs += np.triu(np.ones_like(self.yobs)-self.yobs, 1).T
        self.yobs /= self.nconst
        for i in range(self.n):
            rsum = np.sum(self.yobs[i, :]) - self.yobs[i, i]
            self.yobs[i, i] = 1 - rsum

    def solve(self):
        e, v = la.eig(self.yobs.T)
        idx = np.argmax(e.real)
        self.pi_hat = (v[:, idx].real * np.sign(v[0, idx].real))



# deprecated old version

class LST_deprecated:
    def __init__(self, theta=None, link=sigmoid, n=5, L=1):
        if theta is None:
            self.n = n
            w = 2 * np.random.rand(self.n) - 1
            self.theta = np.sort(w, axis=None)[::-1]
        else:
            self.n = len(theta)
            self.theta = np.sort(theta, axis=None)[::-1]
        self.L = L
        self.link = link
        self.tmat = np.zeros([self.n, self.n])
        self.yobs = np.zeros_like(self.tmat)
        self.nconst = self.n
        for i in range(self.n):
            for j in range(i):
                fij = link(self.theta[j] - self.theta[i])
                self.tmat[i, j] = fij / self.nconst
                self.tmat[j, i] = (1 - fij) / self.nconst
                yij = st.bernoulli.rvs(fij, size=self.L)
                yij = np.mean(yij)
                self.yobs[i, j] = yij / self.nconst
                self.yobs[j, i] = (1 - yij) / self.nconst
        for i in range(self.n):
            rsum = np.sum(self.tmat[i, :])
            self.tmat[i, i] = 1 - rsum
            rsum = np.sum(self.yobs[i, :])
            self.yobs[i, i] = 1 - rsum
        

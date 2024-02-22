from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg as la
import os

save_path = os.path.join('./', 'fig/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(42)

n_list = [8, 16, 32, 64, 128]


F = gaussian_cdf
L = 50

for n in n_list:
    theta = -np.linspace(0, 5, n, endpoint=True)
    theta = theta - np.mean(theta)
    model = LST(theta=theta, link=F, n=n)
    model.sample(L=L, p=2*np.log(n)/n)
    P = model.tmat
    e, v = la.eig(P.T)
    idx = np.argmax(e.real)
    pi = (v[:, idx].real * np.sign(v[0, idx].real))
    pi /= np.sum(pi)

    plt.plot((1+np.arange(n))/n, pi, label="sample size={}".format(n))

plt.title("Fix max-gap=5")
plt.xlabel("quantile")
plt.ylabel("density")
plt.legend()
plt.savefig('./fig/stationary_dist_fixmax.png', dpi=600)
plt.close()


for n in n_list:
    theta = -np.arange(n) * 0.1
    theta = theta - np.mean(theta)
    model = LST(theta=theta, link=F, n=n)
    model.sample(L=L, p=2*np.log(n)/n)
    P = model.tmat
    e, v = la.eig(P.T)
    idx = np.argmax(e.real)
    pi = (v[:, idx].real * np.sign(v[0, idx].real))
    pi /= np.sum(pi)

    plt.plot((1+np.arange(n))/n, pi, label="sample size={}".format(n))

plt.title("Fix min-gap=0.1")
plt.xlabel("quantile")
plt.ylabel("density")
plt.legend()
plt.savefig('./fig/stationary_dist_fixmin.png', dpi=600)

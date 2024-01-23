from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg as la
import os

save_path = os.path.join('./', 'fig/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(42)

L = 1 # number of repeated observations on each pair
rep = 50 # number of repeated random simulations used to calculate the expectation
n_list = [8, 16, 32, 64, 128, 256, 512, 1024] # list of tournament sizes
model_list = ["Bradley-Terry", "Thurstone", "Laplacian CDF"] # list of formal names of the simulated models

mean_dict = {}
sd_dict = {}

for name in model_list:

    F = model_dict[name]
    mean_list = []
    sd_list = []
    for n in n_list:
        theta = -np.arange(n) * 0.1
        theta = theta - np.mean(theta)
        model = LST(theta=theta, link=F, n=n)
        tau = np.zeros(rep)
        for i in tqdm(range(rep)):
            model.sample(L=L)
            model.solve()
            tau[i] = kendall_tau(np.argsort(model.theta), np.argsort(model.pi_hat)) / rep
        mean_list.append(np.mean(tau))
        sd_list.append(np.std(tau))

    mean_dict[name] = mean_list.copy()
    sd_dict[name] = sd_list.copy()


plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.figure(figsize=(10, 6))
plt.suptitle("Fixed min-gap=0.1")
plt.subplot(121)
mean_list = mean_dict["Bradley-Terry"]
sd_list = sd_dict["Bradley-Terry"]
plt.plot(n_list, mean_list, label="Bradley-Terry", marker='.', linewidth=1)
plt.fill_between(n_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
mean_list = mean_dict["Thurstone"]
sd_list = sd_dict["Thurstone"]
plt.plot(n_list, mean_list, label="Thurstone", marker='2', linewidth=1)
plt.fill_between(n_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
mean_list = mean_dict["Laplacian CDF"]
sd_list = sd_dict["Laplacian CDF"]
plt.plot(n_list, mean_list, label="Laplacian CDF", marker='x', linewidth=1)
plt.fill_between(n_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
plt.ylim([0, 0.02])
plt.xlabel(r'$n$')
plt.ylabel(r'Kendall $\tau$')
plt.legend()

plt.subplot(122)
mean_list = mean_dict["Bradley-Terry"]
plt.plot(np.log(n_list), np.log(mean_list), label="Bradley-Terry", marker='.', linewidth=1)
mean_list = mean_dict["Thurstone"]
plt.plot(np.log(n_list), np.log(mean_list), label="Thurstone", marker='2', linewidth=1)
mean_list = mean_dict["Laplacian CDF"]
plt.plot(np.log(n_list), np.log(mean_list), label="Laplacian CDF", marker='x', linewidth=1)
plt.xlabel(r'$\log n$')
plt.ylabel(r'$\log\tau$')
plt.grid(True)
plt.legend()
# plt.axis('scaled')

plt.savefig('./fig/fixmin.png', dpi=800)

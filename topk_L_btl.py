from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg as la
import os

save_path = os.path.join('./', 'fig/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(42)

L = 50 # number of repeated observations on each pair
rep = 100 # number of repeated random simulations used to calculate the expectation
n = 100
L_list = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
model_list = ["Bradley-Terry"] # list of simulated models
delta = 1/3
K = 1

sr_mean_dict = {}
sr_sd_dict = {}

rg_mean_dict = {}
rg_sd_dict = {}

bc_mean_dict = {}
bc_sd_dict = {}

for name in model_list:

    F = model_dict[name]
    theta = np.hstack((np.ones(1), 0.99 * np.ones(1), np.zeros(n-2)))
    model = LST(theta=theta, link=F, n=n)
    sr_mean_list = []
    sr_sd_list = []
    rg_mean_list = []
    rg_sd_list = []
    bc_mean_list = []
    bc_sd_list = []
    for L in L_list:
        sr_tau = np.zeros(rep)
        rg_tau = np.zeros(rep)
        bc_tau = np.zeros(rep)
        for i in tqdm(range(rep)):
            model.sample(L=L, p=0.9)
            model.solve(delta=0)
            sr_tau[i] = topK(np.argsort(-model.theta), np.argsort(-model.pi_hat), K)
            model.solve(delta=delta)
            rg_tau[i] = topK(np.argsort(-model.theta), np.argsort(-model.pi_hat), K)
            model.borda_count()
            bc_tau[i] = topK(np.argsort(-model.theta), np.argsort(-model.borda_score), K)
        sr_mean_list.append(np.mean(sr_tau))
        sr_sd_list.append(np.std(sr_tau))
        rg_mean_list.append(np.mean(rg_tau))
        rg_sd_list.append(np.std(rg_tau))
        bc_mean_list.append(np.mean(bc_tau))
        bc_sd_list.append(np.std(bc_tau))

    sr_mean_dict[name] = sr_mean_list.copy()
    sr_sd_dict[name] = sr_sd_list.copy()
    rg_mean_dict[name] = rg_mean_list.copy()
    rg_sd_dict[name] = rg_sd_list.copy()
    bc_mean_dict[name] = bc_mean_list.copy()
    bc_sd_dict[name] = bc_sd_list.copy()


plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.figure(figsize=(10, 6))
plt.suptitle("Bradley-Terry fixed max-gap=5, "+r'$p=0.9,\ n=100$')
plt.subplot(121)
mean_list = sr_mean_dict["Bradley-Terry"]
sd_list = sr_sd_dict["Bradley-Terry"]
print(mean_list)
plt.plot(L_list, mean_list, label="rank centrality", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
mean_list = rg_mean_dict["Bradley-Terry"]
sd_list = rg_sd_dict["Bradley-Terry"]
plt.plot(L_list, mean_list, label="regularized rank centrality", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
mean_list = bc_mean_dict["Bradley-Terry"]
sd_list = bc_sd_dict["Bradley-Terry"]
plt.plot(L_list, mean_list, label="Borda count", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
plt.ylim([0, 1.2])
plt.axhline(1, color="black", linewidth=1, linestyle='--')
plt.xlabel(r'$L$')
plt.ylabel("top-"+str(K)+" accuracy")
plt.legend()

plt.subplot(122)
mean_list = sr_mean_dict["Bradley-Terry"]
plt.plot(np.log(L_list), mean_list, label="rank centrality", marker='.', linewidth=1)
mean_list = rg_mean_dict["Bradley-Terry"]
plt.plot(np.log(L_list), mean_list, label="regularized rank centrality", marker='.', linewidth=1)
mean_list = bc_mean_dict["Bradley-Terry"]
plt.plot(np.log(L_list), mean_list, label="Borda count", marker='.', linewidth=1)
plt.xlabel(r'$\log L$')
plt.ylabel("top-"+str(K)+" accuracy")
plt.grid(True)
plt.legend()
# plt.axis('scaled')

plt.savefig('./fig/topk_L_btl_fixmax.png', dpi=800)




# sr_mean_dict = {}
# sr_sd_dict = {}

# rg_mean_dict = {}
# rg_sd_dict = {}

# bc_mean_dict = {}
# bc_sd_dict = {}

# for name in model_list:

#     F = model_dict[name]
#     theta = -np.arange(n) * 0.1
#     theta = theta - np.mean(theta)
#     model = LST(theta=theta, link=F, n=n)
#     sr_mean_list = []
#     sr_sd_list = []
#     rg_mean_list = []
#     rg_sd_list = []
#     bc_mean_list = []
#     bc_sd_list = []
#     for n in L_list:
#         sr_tau = np.zeros(rep)
#         bc_tau = np.zeros(rep)
#         for i in tqdm(range(rep)):
#             model.sample(L=L, p=0.95)
#             model.solve(delta=0)
#             sr_tau[i] = topK(np.argsort(model.theta), np.argsort(model.pi_hat), K) / rep
#             model.solve(delta=delta)
#             rg_tau[i] = topK(np.argsort(model.theta), np.argsort(model.pi_hat), K) / rep
#             model.borda_count()
#             bc_tau[i] = topK(np.argsort(model.theta), np.argsort(model.borda_score), K) / rep
#         sr_mean_list.append(np.mean(sr_tau))
#         sr_sd_list.append(np.std(sr_tau))
#         rg_mean_list.append(np.mean(rg_tau))
#         rg_sd_list.append(np.std(rg_tau))
#         bc_mean_list.append(np.mean(bc_tau))
#         bc_sd_list.append(np.std(bc_tau))

#     sr_mean_dict[name] = sr_mean_list.copy()
#     sr_sd_dict[name] = sr_sd_list.copy()
#     rg_mean_dict[name] = rg_mean_list.copy()
#     rg_sd_dict[name] = rg_sd_list.copy()
#     bc_mean_dict[name] = bc_mean_list.copy()
#     bc_sd_dict[name] = bc_sd_list.copy()


# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.figure(figsize=(10, 6))
# plt.suptitle("Bradley-Terry fixed min-gap=0.1, "+r'$p=0.95,\ n=100$')
# plt.subplot(121)
# mean_list = sr_mean_dict["Bradley-Terry"]
# sd_list = sr_sd_dict["Bradley-Terry"]
# plt.plot(L_list, mean_list, label="rank centrality", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
# mean_list = rg_mean_dict["Bradley-Terry"]
# sd_list = rg_sd_dict["Bradley-Terry"]
# plt.plot(L_list, mean_list, label="regularized rank centrality", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
# mean_list = bc_mean_dict["Bradley-Terry"]
# sd_list = bc_sd_dict["Bradley-Terry"]
# plt.plot(L_list, mean_list, label="Borda count", marker='.', linewidth=1)
# plt.fill_between(L_list, np.array(mean_list)-np.array(sd_list), np.array(mean_list)+np.array(sd_list), alpha=0.1)
# # plt.ylim([0, 0.02])
# plt.xlabel(r'$L$')
# plt.ylabel(r'Kendall $\tau$')
# plt.legend()

# plt.subplot(122)
# mean_list = sr_mean_dict["Bradley-Terry"]
# plt.plot(np.log(L_list), mean_list, label="rank centrality", marker='.', linewidth=1)
# mean_list = rg_mean_dict["Bradley-Terry"]
# plt.plot(np.log(L_list), mean_list, label="regularized rank centrality", marker='.', linewidth=1)
# mean_list = bc_mean_dict["Bradley-Terry"]
# plt.plot(np.log(L_list), mean_list, label="Borda count", marker='.', linewidth=1)
# plt.xlabel(r'$\log L$')
# plt.ylabel(r'$\tau$')
# plt.grid(True)
# plt.legend()
# # plt.axis('scaled')

# plt.savefig('./fig/topk_L_btl_fixmin.png', dpi=800)



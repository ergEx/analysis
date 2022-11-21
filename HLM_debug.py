import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import read_hlm_output

save_path = os.path.join(os.path.dirname(__file__), 'HLM_debug_figs')
samples = read_hlm_output(inference_mode = 'parameter_estimation',
                          experiment_version = 'two_gamble_new_c',
                          dataSource = 'simulated_data')

print(samples.keys())
print()
for sample in samples:
    print(f'{sample}: {np.shape(samples[sample])}')
n_conditions = 1
n_chains = 4
n_subjects = 10
n_samples = 50

muLogBetaL=-2.3
muLogBetaU=3.4
sigmaLogBetaL=0.01
sigmaLogBetaU=np.sqrt(((muLogBetaU-muLogBetaL)**2)/12)

muEtaL=-2.5
muEtaU=2.5
sigmaEtaL=0.01
sigmaEtaU=np.sqrt(((muEtaU-muEtaL)**2)/12)

muEtaL= -0.01
muEtaU= 0.01
sigmaEtaL= 0.99
sigmaEtaU= 1.01

#mu eta
mu_eta = np.random.uniform(low=muEtaL, high=muEtaU, size=n_samples)
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['mu_eta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(mu_eta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
plt.tight_layout()
fig.savefig(os.path.join(save_path, f'mu_eta'))
plt.close(fig)

#sigma eta
sigma_eta = np.random.uniform(low=sigmaEtaL, high=sigmaEtaU, size=n_samples)
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['sigma_eta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(sigma_eta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
plt.tight_layout()
fig.savefig(os.path.join(save_path, f'sigma_eta'))
plt.close(fig)

#tau eta
tau_eta =  np.array([float(i)**(-2) for i in sigma_eta])
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['tau_eta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(tau_eta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
plt.tight_layout()
fig.savefig(os.path.join(save_path, f'tau_eta'))
plt.close(fig)

#mu log beta
mu_log_beta = np.random.uniform(low=muLogBetaL, high=muLogBetaU, size=n_samples)
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['mu_log_beta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(mu_log_beta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
plt.tight_layout()
fig.savefig(os.path.join(save_path, f'mu_log_beta'))
plt.close(fig)

#sigma log beta
sigma_log_beta = np.random.uniform(low=sigmaLogBetaL, high=sigmaLogBetaU, size=n_samples)
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['sigma_log_beta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(sigma_log_beta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
    plt.tight_layout()
fig.savefig(os.path.join(save_path, f'sigma_log_beta'))
plt.close(fig)

#tau log beta
tau_log_beta = np.array([float(i)**(-2) for i in sigma_log_beta])
fig, ax = plt.subplots(4, 2)
for c in range(n_conditions):
    for chain in range(n_chains):
        sns.kdeplot(samples['tau_log_beta'][chain,:,c], ax = ax[chain, c], label = 'estimated')
        sns.kdeplot(tau_log_beta, ax = ax[chain,c], label = 'expected')
        ax[chain,c].legend()
plt.tight_layout()
fig.savefig(os.path.join(save_path, f'tau_log_beta'))
plt.close(fig)

#eta
for i in range(n_subjects):
    fig, ax = plt.subplots(4, 2)
    for c in range(n_conditions):
        for chain in range(n_chains):
            sns.kdeplot(samples['eta'][chain,:,i,c], ax = ax[chain, c], label = 'estimated')
            sns.kdeplot(np.random.normal(loc=0, scale=1, size=n_samples), ax = ax[chain, c], label = 'expected')
            ax[chain,c].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f'eta_{i}'))
    plt.close(fig)

#beta
for i in range(n_subjects):
    fig, ax = plt.subplots(4, 2, figsize = (20,20))
    for c in range(n_conditions):
        for chain in range(n_chains):
            sns.kdeplot(samples['beta'][chain,:,i,c], ax = ax[chain, c], label = 'estimated')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f'beta_{i}'))
    plt.close(fig)
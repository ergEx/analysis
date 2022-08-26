import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

root_path = os.path.dirname(__file__)
#READ DATA MISSING!!

#DUMMY DATA
subjects = [1,2]
N = 200
eta = np.ones((2,2,N))
beta = np.ones((2,2,N))

#Additive
mu = 0 # Mean of sample !!! Make sure your data is positive for the lognormal example
sigma = 0.20 # Standard deviation of sample

for i in range(len(subjects)):
    norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
    eta[i,0,:] = norm_dist.rvs(size=N) # Generate sample
    beta[i,0,:] = np.exp(eta[i,0,:])

#Multiplicative
mu = 1 # Mean of sample !!! Make sure your data is positive for the lognormal example
sigma = 0.2 # Standard deviation of sample

for i in range(len(subjects)):
    norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
    eta[i,1,:] = norm_dist.rvs(size=N) # Generate sample
    beta[i,1,:] = np.exp(eta[i,1,:])


n_bins = 100
dynamics = ['Additive','Multiplicative']
colors = ['orange','b']

fig_eta, ax_eta = plt.subplots(nrows=len(set(subjects)),ncols=1,figsize = (12,5))
#fig_beta, ax_beta = plt.subplots(nrows=len(set(subjects)),ncols=1,figsize = (12,5))

df_output = pd.DataFrame(columns=('subject','dynamic','eta_MAP','eta_std','beta_MAP','beta_std'))
for subject_idx, subject in enumerate(subjects):
    subject_dict = dict.fromkeys(list(df_output.columns),None)
    subject_dict['subject'] = subject
    for dynamic_idx, dynamic in enumerate(dynamics):
        subject_dict['dynamic'] = dynamic

        #fit eta (normal distributed)
        fitting_params_eta = scipy.stats.norm.fit(eta[subject_idx,dynamic_idx,:])
        subject_dict['eta_MAP'] = fitting_params_eta[0]
        subject_dict['eta_std'] = fitting_params_eta[1]

        #plot
        fitted_eta = scipy.stats.norm(*fitting_params_eta)
        t = np.linspace(np.min(eta[subject_idx,dynamic_idx,:]), np.max(eta[subject_idx,dynamic_idx,:]), 100)
        sns.histplot(eta[subject_idx,dynamic_idx,:],bins=n_bins,stat="density",color=colors[dynamic_idx], ax=ax_eta[subject_idx],label=dynamic)
        ax_eta[subject_idx].plot(t, fitted_eta.pdf(t), lw=1, color='r')

        ##fit beta (lognormal distributed)
        #fitting_params_beta = scipy.stats.lognorm.fit(beta[dynamic_idx,:], floc=0)
        #subject_dict['beta_MAP'] = fitting_params_beta[2]-fitting_params_beta[0]**2
        #subject_dict['beta_std'] = fitting_params_beta[0]

        ##plot
        #fitted_beta = scipy.stats.lognorm(*fitting_params_beta)
        #t = np.linspace(np.min(beta[dynamic_idx,:]), np.max(beta[dynamic_idx,:]), 100)
        #sns.histplot(beta,bins=n_bins,stat="density", ax=ax_beta[subject_idx])
        #ax_beta[subject_idx].plot(t, fitted_beta.pdf(t), lw=2, color='r')

        df_output = df_output.append(subject_dict, ignore_index=True)

    ax_eta[subject_idx].set_title(f'Subject {subject}')
    ax_eta[subject_idx].legend()
plt.tight_layout()
plt.show()
print(df_output)
df_output.to_csv(os.path.join(root_path, 'data', 'bayesian_model_output.csv'), sep='\t')

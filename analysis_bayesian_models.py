import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

root_path = os.path.dirname(__file__)
#READ DATA MISSING!!

#DUMMY DATA
mu = 0.20 # Mean of sample !!! Make sure your data is positive for the lognormal example
sigma = 0.50 # Standard deviation of sample
N = 2000 # Number of samples
n_bins = 100

norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
eta = norm_dist.rvs(size=N) # Generate sample
etaM = np.exp(eta) # Generate log sample
beta = np.exp(eta)

subjects = [1,2]

fig_subject, ax_subject = plt.subplots(nrows=len(set(subjects)),ncols=3,figsize = (12,5))
print(np.shape(ax_subject))
df_output = pd.DataFrame(columns=('subject','eta_MAP','eta_std','etaM_MAP','etaM_std','beta_MAP','beta_std'))
for subject_idx, subject in enumerate(subjects):
    subject_dict = dict.fromkeys(list(df_output.columns),None)
    subject_dict['subject'] = subject
    #fit eta (mnormal distributed)
    fitting_params_eta = scipy.stats.norm.fit(eta)
    subject_dict['eta_MAP'] = fitting_params_eta[0]
    subject_dict['eta_std'] = fitting_params_eta[1]

    #plot
    fitted_eta = scipy.stats.norm(*fitting_params_eta)
    t = np.linspace(np.min(eta), np.max(eta), 100)
    sns.histplot(eta,bins=n_bins,stat="density", ax=ax_subject[subject_idx,0])
    ax_subject[subject_idx,0].plot(t, fitted_eta.pdf(t), lw=2, color='r')

    #fit etaM (lognormal distributed)
    fitting_params_etaM = scipy.stats.lognorm.fit(etaM, floc=0)
    subject_dict['etaM_MAP'] = fitting_params_etaM[2]-fitting_params_etaM[0]**2
    subject_dict['etaM_std'] = fitting_params_etaM[0]

    #plot
    fitted_etaM = scipy.stats.lognorm(*fitting_params_etaM)
    t = np.linspace(np.min(etaM), np.max(etaM), 100)
    sns.histplot(etaM,bins=n_bins,stat="density", ax=ax_subject[subject_idx,1])
    ax_subject[subject_idx,1].plot(t, fitted_etaM.pdf(t), lw=2, color='r')

    #fit beta (lognormal distributed)
    fitting_params_beta = scipy.stats.lognorm.fit(beta, floc=0)
    subject_dict['beta_MAP'] = fitting_params_beta[2]-fitting_params_beta[0]**2
    subject_dict['beta_std'] = fitting_params_beta[0]

    #plot
    fitted_beta = scipy.stats.lognorm(*fitting_params_beta)
    t = np.linspace(np.min(beta), np.max(beta), 100)
    sns.histplot(beta,bins=n_bins,stat="density", ax=ax_subject[subject_idx,2])
    ax_subject[subject_idx,2].plot(t, fitted_beta.pdf(t), lw=2, color='r')

    df_output = df_output.append(subject_dict, ignore_index=True)
    ax_subject[0,0].set_title('eta estimation')
    ax_subject[0,1].set_title('etaM estimation')
    ax_subject[0,2].set_title('beta estimation')
plt.show()
df_output.to_csv(os.path.join(root_path, 'data', 'bayesian_model_output.csv'), sep='\t')

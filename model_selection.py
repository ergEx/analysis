import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import read_hlm_output

root_path = os.path.dirname(__file__)
design_variant = 'two_gamble_new_c'
inference_mode = 'model_selection'
HLM_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'simulated_data')
print(HLM_samples.keys())

print(np.shape(HLM_samples['etaNoDyn']))
print(np.shape(HLM_samples['etaDyn']))
print(np.shape(HLM_samples['delta_eta']))

n_subjects = 3
n_chains = 4
n_samples = 50

fig, ax = plt.subplots(4,1,figsize=(10,10))
for i in range(n_subjects):
    etaNoDyn_dist = HLM_samples['etaNoDyn'][:,:,i].flatten()
    sns.kdeplot(etaNoDyn_dist.flatten(),ax=ax[0], label = f'Subject {i+1}')
    ax[0].set_title('Eta distribution no dynamic')
    ax[0].legend()

    etaDyn_dist = HLM_samples['etaDyn'][:,:,i]
    sns.kdeplot(etaDyn_dist.flatten(),ax=ax[1], label = f'Subject {i+1}')
    ax[1].set_title('Eta distribution dynamic')
    ax[1].legend()

    delta_eta_dist_mul = HLM_samples['delta_eta'][:,:,i,1]
    sns.kdeplot(delta_eta_dist_mul.flatten(),ax=ax[2], label = f'Subject {i+1}')
    ax[2].set_title('Delta eta distribution multiplicative')
    ax[2].legend()

    delta_eta_dist_add = HLM_samples['delta_eta'][:,:,i,0]
    ax[3].hist(delta_eta_dist_add.flatten())
    ax[3].set_title('Delta eta distribution additive')
fig.tight_layout()

z = HLM_samples['z']

Dyn = 0
Nodyn = 0
z_choices_subject = []
for i in range(n_subjects):
    Dyn_subject = 0
    NoDyn_subject = 0
    for j in range(n_chains):
        for s in range(n_samples):
            if z[j,s,i] in [1,3,5,7]:
                Nodyn += 1
                NoDyn_subject += 1
            else:
                Dyn += 1
                Dyn_subject += 1
    z_choices_subject.append([NoDyn_subject/(n_samples*n_chains),Dyn_subject/(n_samples*n_chains)])
z_choices = [Nodyn/(n_subjects*n_samples*n_chains),Dyn/(n_subjects*n_samples*n_chains)]

print('Model indicator sum over all subjects')
print(z_choices)

print()
print('Model indicator subjectwise')
print(z_choices_subject)


fig_z, ax_z = plt.subplots(figsize=(10,10))
sns.heatmap(z_choices_subject, square=True, ax=ax_z, cmap="binary")
ax_z.set(title="Model_selection",
         yticks=[x + 0.5 for x in list(range(n_subjects))],
         yticklabels=[str(x + 1) for x in list(range(n_subjects))],
         xticklabels=["Dynamic invariant", "Dynamic specific"])

plt.show()
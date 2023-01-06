import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import read_hlm_output

root_path = os.path.dirname(__file__)
design_variant = 'two_gamble_new_c'
inference_mode = 'model_selection'
HLM_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'simulated_data50')
print(HLM_samples.keys())
z = HLM_samples['z']

n_chains, n_samples, n_subjects = np.shape(HLM_samples['z'])

true_param = [0,0.5,1]
color = ['red','blue','black']
linestyle = ['-','--']

fig, ax = plt.subplots(2,1,figsize=(10,10))
for i in range(n_subjects):
    print('sub' , i)
    etaNoDyn_dist = HLM_samples['etaNoDyn'][:,:,i].flatten()
    dist_no_dyn = [x for j, x in enumerate(etaNoDyn_dist) if z[:,:,i].flatten()[j] in [1,3,5,7]]
    print(len(dist_no_dyn))
    sns.kdeplot(dist_no_dyn,ax=ax[0], label = f'Subject {i+1}', color=color[i])
    ax[0].set_title('Eta distribution no dynamic')
    ax[0].legend(fontsize = 'xx-small')
    ax[0].axvline(x=true_param[i],color=color[i])
    ax[0].set_xlim([-2,2])


    for c in range(2):
        print('cond',c)
        etaDyn_dist = HLM_samples['etaDyn'][:,:,i,c].flatten()
        dist_dyn = [x for j, x in enumerate(etaDyn_dist) if z[:,:,i].flatten()[j] in [2,4,6,8]]
        print(len(dist_dyn))
        sns.kdeplot(dist_dyn,ax=ax[1], label = f'Subject {i+1}, condition {c}', color=color[i], linestyle = linestyle[c])
        ax[1].set_title('Eta distribution dynamic')
        ax[1].legend(fontsize = 'xx-small')
    ax[1].axvline(x=true_param[i],color=color[i])
    ax[1].set_xlim([-2,2])
#    delta_eta_dist_mul = HLM_samples['delta_eta'][:,:,i,1]
#    sns.kdeplot(delta_eta_dist_mul.flatten(),ax=ax[2], label = f'Subject {i+1}')
#    ax[2].set_title('Delta eta distribution multiplicative')
#    ax[2].legend(fontsize = 'xx-small' )
#
#    delta_eta_dist_add = HLM_samples['delta_eta'][:,:,i,0]
#    ax[3].hist(delta_eta_dist_add.flatten())
#    ax[3].set_title('Delta eta distribution additive')
fig.tight_layout()



z_dyn = []

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
sns.heatmap(z_choices_subject, square=False, ax=ax_z, cmap="binary")
ax_z.set(title="Model Selection",
         yticklabels=[str(x + 1) for x in list(range(n_subjects))],
         xticklabels=["Dynamic invariant", "Dynamic specific"])
fig_z.tight_layout()
plt.show()
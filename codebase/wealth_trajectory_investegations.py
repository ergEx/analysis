#%% # -*- coding: utf-8 -*-

#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import isoelastic_utility, wealth_change

#%%
#read in data
data_dir = os.path.join('..','data','1_pilot')
df = pd.read_csv(os.path.join(data_dir, 'all_active_phase_data.csv', sep = '\t'))
print(f'Total number of rows: {len(df)}')
print(f'Number of participants in data: {len(set(df.participant_id))}')
print(f'Number of conditions in data: {len(set(df.eta))}')
print(f'Number of Trials per participant per condition: {int(len(df)/len(set(df.participant_id))/len(set(df.eta)))}')
print(f'Columns:')
for col in df.columns:
    print(f'- {col}')

#%%
#Add data to dataframe
def add_data(group):
    group = group.sort_values(by='trial')

    #expected_gamma_opt
    expected_gamma_opt = np.insert(np.ones(len(group)), 0, 1000)

    #expected_gamma_sub
    expected_gamma_sub = np.insert(np.ones(len(group)), 0, 1000)

    #gamma_opt
    gamma_opt = np.insert(np.ones(len(group)), 0, 1000)

    #gamma_opt
    gamma_sub = np.insert(np.ones(len(group)), 0, 1000)

    #sub_etas
    sub_etas = np.insert(np.ones((len(group), 5)), 0, 1000, axis=0)

    for idx, row in group.iterrows():
        t = row['trial']
        g1 = row['gamma_left_up']
        g2 = row['gamma_left_down']
        g3 = row['gamma_right_up']
        g4 = row['gamma_right_down']

        #expected_gamma_opt
        if (g1 + g2) / 2 > (g3 + g4) / 2:
            expected_gamma_opt[t+1] = wealth_change(expected_gamma_opt[t], (g1 + g2) /2, row['eta'])
        else:
            expected_gamma_opt[t+1] = wealth_change(expected_gamma_opt[t], (g3 + g4)/2, row['eta'])

        #expected_gamma_sub
        if row['selected_side'] == 'left':
            expected_gamma_sub[t+1] = wealth_change(expected_gamma_sub[t], (g1 + g2) /2, row['eta'])
        else:
            expected_gamma_sub[t+1] = wealth_change(expected_gamma_sub[t], (g3 + g4)/2, row['eta'])

        #gamma_opt
        if (g1 + g2) / 2 > (g3 + g4) / 2:
            if row['gamble_up'] == 'up':
                gamma_opt[t+1] = wealth_change(gamma_opt[t], g1, row['eta'])
            else:
                gamma_opt[t+1] = wealth_change(gamma_opt[t], g2, row['eta'])
        else:
            if row['gamble_up'] == 'up':
                gamma_opt[t+1] = wealth_change(gamma_opt[t], g3, row['eta'])
            else:
                gamma_opt[t+1] = wealth_change(gamma_opt[t], g4, row['eta'])

        #gamma_sub
        if row['selected_side'] == 'left':
            if row['gamble_up'] == 'up':
                gamma_sub[t+1] = wealth_change(gamma_sub[t], g1, row['eta'])
            else:
                gamma_sub[t+1] = wealth_change(gamma_sub[t], g2, row['eta'])
        else:
            if row['gamble_up'] == 'up':
                gamma_sub[t+1] = wealth_change(gamma_sub[t], g3, row['eta'])
            else:
                gamma_sub[t+1] = wealth_change(gamma_sub[t], g4, row['eta'])

        for i, sub_eta in enumerate([-0.5,0.0,0.5,1.0,1.5]):
            if min(sub_etas[t,i],
                   wealth_change(sub_etas[t,i], g1, row['eta']),
                   wealth_change(sub_etas[t,i], g1, row['eta']),
                   wealth_change(sub_etas[t,i], g1, row['eta']),
                   wealth_change(sub_etas[t,i], g1, row['eta'])) > 0:
                choice = (((isoelastic_utility(wealth_change(sub_etas[t,i], g1, row['eta']), sub_eta) +
                     isoelastic_utility(wealth_change(sub_etas[t,i], g2, row['eta']), sub_eta)) / 2)
                   >
                   ((isoelastic_utility(wealth_change(sub_etas[t,i], g3, row['eta']), sub_eta) +
                     isoelastic_utility(wealth_change(sub_etas[t,i], g4, row['eta']), sub_eta)) / 2))
            else:
                choice = (g1 + g2) / 2 > (g3 + g4) / 2

            if choice:
                if row['gamble_up'] == 'up':
                    sub_etas[t+1,i] = wealth_change(sub_etas[t,i], g1, row['eta'])
                else:
                    sub_etas[t+1,i] = wealth_change(sub_etas[t,i], g2, row['eta'])
            else:
                if row['gamble_up'] == 'up':
                    sub_etas[t+1,i] = wealth_change(sub_etas[t,i], g3, row['eta'])
                else:
                    sub_etas[t+1,i] = wealth_change(sub_etas[t,i], g4, row['eta'])




    group['expected_gamma_opt'] = expected_gamma_opt[:-1]
    group['expected_gamma_sub'] = expected_gamma_sub[:-1]
    group['gamma_opt'] = gamma_opt[:-1]
    group['gamma_sub'] = gamma_sub[:-1]
    group['n0_5'] = sub_etas[:-1,0]
    group['0_0'] = sub_etas[:-1,1]
    group['0_5'] = sub_etas[:-1,2]
    group['1_0'] = sub_etas[:-1,3]
    group['1_5'] = sub_etas[:-1,4]

    return group

df = df.groupby(['participant_id', 'eta']).apply(add_data).reset_index(drop=True)

#%%
#plot trajectories of descriptors per participant
labels = ['wealth', 'expected_gamma_opt', 'expected_gamma_sub', 'gamma_opt', 'gamma_sub']
etas = [0.0, 1.0]
N = 3
fig, ax = plt.subplots(N, 2, figsize=(15, 3*N))
for j, participant in enumerate(list(set(df.participant_id))[:N]):
    ax[j,0].set_title(participant)
    for i, eta in enumerate(etas):
        df_eta = df.query('participant_id == @participant and eta == @eta')

        for label in labels:
            ax[j,i].plot(df_eta['trial'], df_eta[label], label=label)

        ax[j,i].legend()

    ax[j,1].set_yscale('log')
fig.tight_layout()
plt.show()

#%%
#Calculate and save specific values based on end values
tmp = {0.0: {'wealth': [], 'expected': [], 'sub': []} ,
       1.0: {'wealth': [], 'expected': [], 'sub': []}}
for j, participant in enumerate(list(set(df.participant_id))[:1]):
    for i, eta in enumerate(tmp.keys()):
        df_eta = df.query('participant_id == @participant and eta == @eta')
        tmp[eta]['wealth'].append(df_eta['wealth'].iloc[-1])
        tmp[eta]['expected'].append(df_eta['expected_gamma_opt'].iloc[-1] - df_eta['expected_gamma_sub'].iloc[-1])
        tmp[eta]['sub'].append(df_eta['expected_gamma_sub'].iloc[-1] - df_eta['wealth'].iloc[-1])

tmp[1.0] = {key: [np.log(value) for value in values] for key, values in tmp[1.0].items()}

#%%
#check hypothetical wealth trajectories based on the experienced realizations on the experiment
custom_palette = sns.color_palette("Set1", n_colors=2)

etas = [0.0, 1.0]
fig, ax = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]})

sub_eta = ['n0_5', '0_0', '0_5', '1_0', '1_5']
sub_eta = ['0_0','1_0']
for eta_idx, eta in enumerate(etas):
    log_scale = True if eta == 1.0 else False
    densities = {key: [] for key in sub_eta}

    for participant in list(set(df.participant_id)):# df['participant_id'].unique():
        df_eta = df.query('participant_id == @participant and eta == @eta')
        for l, label in enumerate(sub_eta):
            ax[eta_idx,0].plot(df_eta['trial'], df_eta[label], label=label, color=custom_palette[l])
            densities[label].append(df_eta[label].iloc[-1])

    densities['difference'] = [x - y for x, y in zip(densities['0_0'], densities['1_0'])] if eta == 0.0 else [np.log(x) - np.log(y) for x, y in zip(densities['1_0'], densities['0_0'])]


    # Plot the right plot (density plot)
    #for i, (key, values) in enumerate(densities.items()):
    #sns.kdeplot(data=values, color=custom_palette[i], label=key, ax=ax[eta_idx,1], log_scale=log_scale)
    #sns.kdeplot(data=densities['subtract'], color=custom_palette[0], label=key, ax=ax[eta_idx,1], log_scale=False)
    ax[eta_idx, 1].hist(densities['difference'])
    ax[eta_idx, 1].axvline(0)

ax[1,0].set(yscale = 'log')
plt.show()

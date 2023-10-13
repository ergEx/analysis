#%% # -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
from utils import isoelastic_utility, wealth_change

#%%
#read in data
df = pd.read_csv('all_active_phase_data.csv', sep = '\t')
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
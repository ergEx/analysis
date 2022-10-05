import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESET = 45
root_path = os.path.dirname(__file__)
design_variant = 'test'

condition_specs = {0.0:'Additive', 1.0:'Multiplicative'}

passive_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_passive_phase_data.csv'), sep='\t')
active_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_active_phase_data.csv'), sep='\t')

payment_df = pd.DataFrame(columns=['Participant','end_wealth'])

for c,condition in enumerate(set(passive_phase_df['eta'])):
    for i,subject in enumerate(set(passive_phase_df['participant_id'])):
        save_path = os.path.join(root_path, 'figs', design_variant, str(subject))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        '''PASIVE PHASE'''
        passive_subject_df = passive_phase_df.query('participant_id == @subject and eta == @condition').reset_index(drop=True)
        fig, ax = plt.subplots(1,1)
        ax.plot(passive_subject_df.trial, passive_subject_df.wealth)
        ax.set(title=f'Passive phase \nSubject {subject}, Condition: {condition_specs[condition]}',
               xlabel='Trial',
               ylabel='Wealth')
        for reset_idx in range(1,4):
            ax.axvline(x=RESET*reset_idx,color='grey',linestyle='--')
        ax.plot([], label="Reset",color='grey',linestyle='--')
        ax.legend()

        fig.savefig(os.path.join(save_path, f'Passive_trajectory_{condition_specs[condition]}.png'))
        plt.close(fig)

        '''ACTIVE PHASE'''
        active_subject_df = active_phase_df.query('participant_id == @subject and eta == @condition').reset_index(drop=True)


        #Indifference eta plots
        fig, ax = plt.subplots(1,1)
        for ii, choice in enumerate(active_subject_df['selected_side_map']):
            trial = active_subject_df.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue

            ax.axvline(condition, linestyle='--', linewidth=1, color='k')

            ax.plot(trial.indif_eta, ii, marker=trial.min_max_sign, color=trial.min_max_color)

            ax.set(title = f'Indifference eta \nSubject {subject}, Condition: {condition_specs[condition]}',
                   xlabel = 'Riskaversion ($\eta$)')

            ax.axes.yaxis.set_visible(False)

        fig.savefig(os.path.join(save_path, f'Indifference_eta_{condition_specs[condition]}.png'))
        ax.set_xlim([-2,3])
        fig.savefig(os.path.join(save_path, f'Indifference_eta_zoom_{condition_specs[condition]}.png'))
        plt.close(fig)



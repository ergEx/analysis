import os
import sys

import matplotlib.pyplot as plt
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
        ax.set(title=f'Passive phase, Subject {subject}, Condition: {condition_specs[condition]}',
               xlabel='Trial',
               ylabel='Wealth')
        for reset_idx in range(1,4):
            ax.axvline(x=RESET*reset_idx,color='grey',linestyle='--')
        ax.plot([], label="Reset",color='grey',linestyle='--')
        ax.legend()

        plt.savefig(os.path.join(save_path, f'Passive_trajectory_{condition_specs[condition]}.png'))


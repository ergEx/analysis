###read data and save data

import os

import numpy as np
import pandas as pd
import scipy.io

import utils

root_path = os.path.join(os.path.dirname(__file__),)
subject_ids = [0,1]
dynamics = ['Additive','Multiplicative']
run = 1

#CSV
df = pd.DataFrame()
for dynamic in dynamics:
    dynamic_text = {'Additive': '0d0', 'Multiplicative': '1d0'}
    for subject in subject_ids:
        data = pd.read_csv(os.path.join(root_path, 'data','experiment_output',f'sub-{subject}_ses-lambd{dynamic_text[dynamic]}_task-active_run-{run}_events.tsv'),sep='\t')
        subject_df = data.query('event_type == "WealthUpdate"').reset_index(drop=True)

        subject_df = utils.add_info_to_df(subject_df, subject)
        df = pd.concat([df, subject_df])
df.to_csv(os.path.join(root_path, 'data', 'all_data.csv'), sep='\t')

#.mat
dynamic_specs = {'Additive': {'text': '0d0', 'txt_append': '_add'},
                 'Multiplicative': {'text': '1d0', 'txt_append': '_mul'}}
datadict = dict()
for dynamic in dynamics:
    for subject in subject_ids:
        data = pd.read_csv(os.path.join(root_path, 'data','experiment_output',f'sub-{subject}_ses-lambd{dynamic_specs[dynamic]["text"]}_task-active_run-{run}_events.tsv'),sep='\t')
        subject_df = data.query('event_type == "WealthUpdate"').reset_index(drop=True)
        subject_df = utils.add_info_to_df(subject_df, subject)

        '''
        Append which session first (has not been decided yet)
        '''
        if subject > 500:
            datadict.setdefault('MultiplicativeSessionFirst',[]).append(1)
        else:
            datadict.setdefault('MultiplicativeSessionFirst',[]).append(0)

        '''
        Retrieve growth rates
        '''
        datadict.setdefault('gr1_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_left_up']))
        datadict.setdefault('gr1_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_left_down']))
        datadict.setdefault('gr2_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_right_up']))
        datadict.setdefault('gr2_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_right_down']))

        '''
        Retrieve wealth changes
        '''
        datadict.setdefault('x1_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['x1_1']))
        datadict.setdefault('x1_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['x1_2']))
        datadict.setdefault('x2_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['x2_1']))
        datadict.setdefault('x2_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['x2_2']))

        '''
        Retrive wealth
        '''
        datadict.setdefault('wealth'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['wealth']))

        '''
        Retrieve keypresses
        '''
        datadict.setdefault('choice'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['selected_side_map']))
scipy.io.savemat(os.path.join(os.path.join(root_path,'data','all_data.mat')),datadict,oned_as='row')
np.savez(os.path.join(os.path.join(root_path,'data','all_data.mat.npz')),datadict = datadict)

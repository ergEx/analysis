###read data and save data

import os

import numpy as np
import pandas as pd
import scipy.io
import sub_specs
import utils

simulation = True
root_path = os.path.join(os.path.dirname(__file__),)
design_variant = 'two_gamble_new_c'
condition_specs = {'condition':['Additive','Multiplicative'], 'lambd':[0.0,1.0], 'bids_text': ['0d0','1d0'],'txt_append':['_add','_mul']}
if simulation:
    subject_specs = {'id':list(range(10)), 'simulation_eta': [0.0,0.5,1.]}
else:
    subject_specs = sub_specs.sub_specs(design_variant)
n_passive_runs = 3

passive_phase_df = pd.DataFrame()
active_phase_df = pd.DataFrame()
datadict = dict()
for c,condition in enumerate(condition_specs['condition']):
    for i,subject in enumerate(subject_specs['id']):
        for j, eta in enumerate(subject_specs['simulation_eta']):
            print(c,i,j)
            if not simulation:
                '''Passive phase data'''
                for run in range(1,n_passive_runs):
                    passive_phase_data = pd.read_csv(os.path.join(root_path, 'data',design_variant,f'sub-{subject}',f'ses-{subject_specs["first_run"][i][c]}',f'sub-{subject}_ses-{subject_specs["first_run"][i][c]}_task-passive_acq-lambd{condition_specs["bids_text"][c]}_run-{run}_beh.csv'),sep='\t')

                    subject_df = passive_phase_data.query('event_type == "WealthUpdate"').reset_index(drop=True)
                    subject_df = subject_df.query('part == 0').reset_index(drop=True)
                    passive_phase_df = pd.concat([passive_phase_df, subject_df])

            '''Active phase data'''
            run = 1
            if simulation:
                subject_df = pd.read_csv(os.path.join(root_path, 'data', design_variant, 'simulations', f'sim_agent_{subject}_lambd_{condition_specs["lambd"][c]}_{eta}.csv' ), sep='\t')
            else:
                active_phase_data = pd.read_csv(os.path.join(root_path, 'data',design_variant,f'sub-{subject}',f'ses-{subject_specs["first_run"][i][c]}',f'sub-{subject}_ses-{subject_specs["first_run"][i][c]}_task-active_acq-lambd{condition_specs["bids_text"][c]}_run-{run}_beh.csv'),sep='\t')
                subject_df = active_phase_data.query('event_type == "WealthUpdate"').reset_index(drop=True)

            subject_df = utils.add_info_to_df(subject_df, subject)

            ##CSV
            active_phase_df = pd.concat([active_phase_df, subject_df])

            ##.mat
            subject_df.loc[np.isnan(subject_df['indif_eta']), 'selected_side_map'] = np.nan

            #Retrieve growth rates
            datadict.setdefault(f'gr1_1{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['gamma_left_up']))
            datadict.setdefault(f'gr1_2{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['gamma_left_down']))
            datadict.setdefault(f'gr2_1{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['gamma_right_up']))
            datadict.setdefault(f'gr2_2{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['gamma_right_down']))

            #Retrieve wealth changes
            datadict.setdefault(f'x1_1{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['x1_1']))
            datadict.setdefault(f'x1_2{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['x1_2']))
            datadict.setdefault(f'x2_1{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['x2_1']))
            datadict.setdefault(f'x2_2{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['x2_2']))

            #Retrive wealth
            datadict.setdefault(f'wealth{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['wealth']))

            #Retrieve keypresses
            datadict.setdefault(f'choice{condition_specs["txt_append"][c]}',[]).append(np.array(subject_df['selected_side_map']))

if simulation:
    simulation_path = os.path.join(root_path, 'data',design_variant,'simulations')
    if not os.path.isdir(simulation_path):
        os.makedirs(simulation_path)
    active_phase_df.to_csv(os.path.join(simulation_path,'all_active_phase_data.csv'),sep='\t')
    scipy.io.savemat(os.path.join(os.path.join(simulation_path,'all_active_phase_data.mat')),datadict,oned_as='row')
    np.savez(os.path.join(os.path.join(simulation_path,'all_active_phase_data.mat.npz')),datadict = datadict)
else:
    passive_phase_df.to_csv(os.path.join(root_path,'data',design_variant,'all_passive_phase_data.csv'),sep='\t')
    active_phase_df.to_csv(os.path.join(root_path,'data',design_variant,'all_active_phase_data.csv'),sep='\t')
    scipy.io.savemat(os.path.join(os.path.join(root_path,'data',design_variant,'all_active_phase_data.mat')),datadict,oned_as='row')
    np.savez(os.path.join(os.path.join(root_path,'data',design_variant,'all_active_phase_data.mat.npz')),datadict = datadict)


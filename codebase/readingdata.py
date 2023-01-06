import os

import numpy as np
import pandas as pd
import scipy.io

from .experiment_specs import condition_specs, sub_specs
from .utils import add_info_to_df


def reading_data(simulation:bool, data_variant:str, n_passive_runs:int) -> None:
    """
    Reads in passive and active phase data for a given design variant, simulation status, and number of passive runs. It stores a tuple of two dataframes, one for passive phase data and one for active phase data. It also stores .mat and .npz versions of the active phase data in a subdirectory within the data directory.

    Parameters:
    simulation (bool): Whether to read in data from simulations or real experiments.
    design_variant (str): The design variant of the experiment.
    n_passive_runs (int): The number of passive runs in the experiment.
    Returns:

    one. Data is stored in the relevant subdirectories within the data directory.
    """
    print('READING DATA')

    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(data_variant)

    ROOT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', data_variant)

    passive_phase_df = pd.DataFrame()
    active_phase_df = pd.DataFrame()
    datadict = dict()
    for c,condition in enumerate(CONDITION_SPECS['condition']):
        print(f'Condition {c+1} of {len(CONDITION_SPECS["lambd"])}')
        for i,subject in enumerate(SUBJECT_SPECS['id']):
            print(f'Subject {i+1} of {len(SUBJECT_SPECS["id"])}')
            '''Passive phase data'''
            if not simulation:
                for run in range(1,n_passive_runs+1):
                    passive_phase_data = pd.read_csv(os.path.join(ROOT_PATH,f'sub-{subject}',f'ses-{SUBJECT_SPECS["first_run"][i][c]}',f'sub-{subject}_ses-{SUBJECT_SPECS["first_run"][i][c]}_task-passive_acq-lambd{CONDITION_SPECS["bids_text"][c]}_run-{run}_beh.csv'),sep='\t')

                    subject_df = passive_phase_data.query('event_type == "WealthUpdate"').reset_index(drop=True)
                    subject_df = subject_df.query('part == 0').reset_index(drop=True)
                    passive_phase_df = pd.concat([passive_phase_df, subject_df])

            '''Active phase data'''
            run = 1
            if simulation:
                subject_df = pd.read_csv(os.path.join(ROOT_PATH,f'sim_agent_{subject}_lambd_{int(CONDITION_SPECS["lambd"][c])}.csv'), sep='\t')
            else:
                active_phase_data = pd.read_csv(os.path.join(ROOT_PATH,f'sub-{subject}',f'ses-{SUBJECT_SPECS["first_run"][i][c]}',f'sub-{subject}_ses-{SUBJECT_SPECS["first_run"][i][c]}_task-active_acq-lambd{CONDITION_SPECS["bids_text"][c]}_run-{run}_beh.csv'),sep='\t')
                subject_df = active_phase_data.query('event_type == "WealthUpdate"').reset_index(drop=True)

            subject_df = add_info_to_df(subject_df)

            ##CSV
            active_phase_df = pd.concat([active_phase_df, subject_df])

            ##.mat
            subject_df.loc[np.isnan(subject_df['indif_eta']), 'selected_side_map'] = np.nan

            #Retrieve growth rates
            datadict.setdefault(f'gr1_1{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['gamma_left_up']))
            datadict.setdefault(f'gr1_2{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['gamma_left_down']))
            datadict.setdefault(f'gr2_1{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['gamma_right_up']))
            datadict.setdefault(f'gr2_2{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['gamma_right_down']))

            #Retrieve wealth changes
            datadict.setdefault(f'x1_1{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['x1_1']))
            datadict.setdefault(f'x1_2{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['x1_2']))
            datadict.setdefault(f'x2_1{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['x2_1']))
            datadict.setdefault(f'x2_2{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['x2_2']))

            #Retrive wealth
            datadict.setdefault(f'wealth{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['wealth']))

            #Retrieve keypresses
            datadict.setdefault(f'choice{CONDITION_SPECS["txt_append"][c]}',[]).append(np.array(subject_df['selected_side_map']))



        active_phase_df.to_csv(os.path.join(ROOT_PATH, 'all_active_phase_data.csv'), sep='\t')
        scipy.io.savemat(os.path.join(ROOT_PATH, 'all_active_phase_data.mat'), datadict, oned_as='row')
        np.savez(os.path.join(ROOT_PATH, 'all_active_phase_data.mat.npz'), datadict=datadict)

        if passive_phase_df is not None:
            passive_phase_df.to_csv(os.path.join(ROOT_PATH, 'all_passive_phase_data.csv'), sep='\t')


if __name__=='__main__':
    simulation = True
    data_variant = '0_simulation'
    n_passive_runs = 3

    reading_data(simulation, data_variant, n_passive_runs)
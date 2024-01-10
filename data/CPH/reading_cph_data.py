import os

import numpy as np
import pandas as pd
from scipy import io


DATA_PATH = ''

PARTICIPANTS = ['Subj01', 'Subj02', 'Subj03', 'Subj05', 'Subj06', 'Subj07', 'Subj08', 'Subj09',
                'Subj10', 'Subj11', 'Subj12' ,'Subj13', 'Subj14', 'Subj15', 'Subj16', 'Subj17',
                'Subj18', 'Subj19', 'Subj20']

dfs = []

selected_columns = ['subjID', 'earnings', 'Gam1_1', 'Gam1_2', 'Gam2_1', 'Gam2_2', 'KP_Final', 'eta']
renamed_columns = {'subjID': 'participant_id', 'earnings': 'wealth',
                   'Gam1_1': 'x1_1', 'Gam1_2': 'x1_2', 'Gam2_1': 'x2_1', 'Gam2_2': 'x2_2', 'KP_Final': 'selected_side', 'eta': 'eta'}
value_mapping = {8.0: 0, 9.0: 1, np.nan: np.nan}
datadict = {}

condition_map = {'add': 'Additive', 'mul': 'Mutiplicative'}

min_n_trials = 299

for condition in ['add', 'mul']:
    for sub in PARTICIPANTS:

        file = os.path.join(DATA_PATH, sub, condition_map[condition],
                            'Data_active', f'{sub[-2:]}_2.txt')

        if not os.path.isfile(file) and condition == 'mul':
            file = os.path.join(DATA_PATH, sub, 'Multiplicative',
                            'Data_active', f'{sub[-2:]}_2.txt')

        try:
            df = pd.read_csv(file, delimiter='\t', header=0, error_bad_lines=True,
                             usecols=range(0, 27))
        except ValueError:
            df = pd.read_csv(file, delimiter='\t', header=0, error_bad_lines=True,
                             usecols=range(0, 25))

        df['eta'] = 1 if condition == 'mul' else 0
        df = df[selected_columns].rename(columns=renamed_columns)

        df['choice'] = df['selected_side'].map(value_mapping)
        df = df[:min_n_trials]

        dfs.append(df)

        for key in ['x1_1', 'x1_2', 'x2_1', 'x2_2', 'wealth', 'choice']:
            datadict.setdefault(f'{key}_{condition}', []).append(
            np.array(df[key])
        )

combined_df = pd.concat(dfs, ignore_index=True)

combined_df.to_csv("all_active_phase_data.csv", sep="\t")
io.savemat("all_active_phase_data.mat", datadict, oned_as="row")
np.savez("all_active_phase_data.mat.npz", datadict=datadict)

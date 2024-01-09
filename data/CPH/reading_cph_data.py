import os

import numpy as np
import pandas as pd
from scipy import io

dfs = []

selected_columns = ['subjID', 'earnings', 'Gam1_1', 'Gam1_2', 'Gam2_1', 'Gam2_2', 'KP_Final', 'eta']
renamed_columns = {'subjID': 'participant_id', 'earnings': 'wealth',
                   'Gam1_1': 'x1_1', 'Gam1_2': 'x1_2', 'Gam2_1': 'x2_1', 'Gam2_2': 'x2_2', 'KP_Final': 'selected_side', 'eta': 'eta'}
value_mapping = {8.0: 0, 9.0: 1, np.nan: np.nan}
datadict = {}

excluded_participant = 5
min_n_trials = 299

for condition in ['add', 'mul']:
    files = [file for file in os.listdir(condition) if file.endswith(".txt")]

    for file in files:
        file_path = os.path.join(condition, file)
        df = pd.read_csv(file_path, delimiter='\t')

        if df.subjID.unique()[0] == excluded_participant:
            continue
        else:

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

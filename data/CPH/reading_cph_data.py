import os
import sys
sys.path.append(os.path.abspath("../../"))

import numpy as np
import pandas as pd
from scipy import io

from codebase.utils import isoelastic_utility, wealth_change

DATA_PATH = 'https://raw.githubusercontent.com/ollie-hulme/ergodicity-breaking-choice-experiment/master/data/'

PARTICIPANTS = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11',
                '12', '13', '14', '15', '16', '17', '18', '19']
# Subject 5 was excluded due to information on https://github.com/ollie-hulme/ergodicity-breaking-choice-experiment/tree/master

dfs = []

selected_columns = ['subjID', 'earnings', 'Gam1_1', 'Gam1_2', 'Gam2_1', 'Gam2_2', 'KP_Final', 'eta']
renamed_columns = {'subjID': 'participant_id', 'earnings': 'wealth',
                   'Gam1_1': 'Gam1_1', 'Gam1_2': 'Gam1_2', 'Gam2_1': 'Gam2_1', 'Gam2_2': 'Gam2_2', 'KP_Final': 'selected_side', 'eta': 'eta'}
value_mapping = {8.0: 0, 9.0: 1, np.nan: np.nan}
redo = {0: 1.0, 1: 100.0}# The growth factors in the txt files are multiplied with a factor of 100 (e.g. 183.02 instead of 1.8302), this list will be used to correct for this
datadict = {}

condition_map = {'add': 'TxtFiles_additive', 'mul': 'TxtFiles_multiplicative'}

min_n_trials = 299

for condition in ['add', 'mul']:
    for sub in PARTICIPANTS:

        file = os.path.join(DATA_PATH, condition_map[condition], f'{sub}_2.txt')
        print(file)
        df = pd.read_csv(file, delimiter='\t')

        df['eta'] = 1 if condition == 'mul' else 0
        df = df[selected_columns].rename(columns=renamed_columns)

        df['choice'] = df['selected_side'].map(value_mapping)

        df['gr1_1'] = isoelastic_utility(df['Gam1_1']/redo[1],1) if condition == 'mul' else df['Gam1_1']/redo[0]
        df['gr1_2'] = isoelastic_utility(df['Gam1_2']/redo[1],1) if condition == 'mul' else df['Gam1_2']/redo[0]
        df['gr2_1'] = isoelastic_utility(df['Gam2_1']/redo[1],1) if condition == 'mul' else df['Gam2_1']/redo[0]
        df['gr2_2'] = isoelastic_utility(df['Gam2_2']/redo[1],1) if condition == 'mul' else df['Gam2_2']/redo[0]

        df['x1_1'] = wealth_change(df['wealth'], df['gr1_1'], df['eta'][0]) - df['wealth']
        df['x1_2'] = wealth_change(df['wealth'], df['gr1_2'], df['eta'][0]) - df['wealth']
        df['x2_1'] = wealth_change(df['wealth'], df['gr2_1'], df['eta'][0]) - df['wealth']
        df['x2_2'] = wealth_change(df['wealth'], df['gr2_2'], df['eta'][0]) - df['wealth']

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

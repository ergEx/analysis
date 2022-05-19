import numpy as np
import pandas as pd
import os
from scipy import misc



def read_active_data(root_path:str, dynamics:list, n_subjects:int, choice_dict:dict = {'right': 1, 'left': 0}):
   print('Reading active data - not fully implemented')
   # dfs = dict()
   # for dynamic in dynamics:
   #     dynamic_dfs = dict()
   #     for subject in range(n_subjects):
   #         data = pd.read_csv(os.path.join(root_path, dynamic, subject),sep='\t') #not done with path
   #         subject_df = data[data['Stream'] == 'Math'].reset_index() #not done with values

   #         for column in ['Name','Name2']: #change column names
   #             subject_df[column] = subject_df[column].map(choice_dict)
   #
   #         dynamic_dfs[subject] = subject_df
   #     dfs[dynamic] = dynamic_dfs
   # return dfs

def indiference_eta(x1:float, x2:float, x3:float, x4:float, w:float, left:int) -> list:
    if w+x1<0 or w+x2<0 or w+x3<0 or w+x4<0:
        '''
        CHANGE THIS CHECK TO JUST SKIP TRIALS WHERE THIS HAPPENS, AS WE ALLOW FOR NEGATIVE WEALTH (KIN ADDITIVE)
        '''
        raise ValueError(f"Isoelastic utility function not defined for negative values")

    func = lambda x : (((((w+x1)**(1-x))/(1-x) + ((w+x2)**(1-x))/(1-x))/2 - ((w)**(1-x))/(1-x))
                    - ((((w+x3)**(1-x))/(1-x) + ((w+x4)**(1-x))/(1-x))/2 - ((w)**(1-x))/(1-x)) )
    for i,x in enumerate(np.linspace(left,100,1000)):
        if x == 1:
            continue
        tmp = func(x)
        if i == 0 or np.sign(prev) == np.sign(tmp):
            prev = tmp
        else:
            return x, func

def calculate_min_v_max(root:float, func, choice:int) -> dict:
    dx = misc.derivative(func,root)
    if dx<0:
        return {'color':'orange','sign':'>', 'val':'max'} if choice==0 else {'color':'b','sign':'<', 'val':'min'}
    else:
        return {'color':'orange','sign':'>', 'val':'max'} if choice==1 else {'color':'b','sign':'<', 'val':'min'}

def is_statewise_dominated(gamble_pair) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater_equal(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater_equal(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater_equal(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater_equal(min(gamble_pair[1]), min(gamble_pair[0])) )


def plot_indifference_eta(choices:list[list],x1:list[list],x2:list[list],x3:list[list],x4:list[list], w:list[list], filenames:list, ax, dynamic:str, left:int) -> None:
    tick_place = [0]
    tick_names = []
    trial_counter = 0
    for subject in range(len(filenames)):
        trial_counter_participant = 0
        if subject == 4: #Participant who did not meet the threshold of correct nobrainers
            continue
        else:
            tick_names.append(f'Participant {subject + 1}')
        statewise_dominated_counter = 0
        for ii, choice in enumerate(choices[subject]):
            if is_statewise_dominated([[x1[subject][ii], x2[subject][ii]],[x3[subject][ii], x4[subject][ii]]]):
                statewise_dominated_counter += 1
                continue
            trial_counter += 1
            trial_counter_participant += 1
            root_dyn, func_dyn = indiference_eta(x1[subject][ii], x2[subject][ii], x3[subject][ii], x4[subject][ii], w[subject][ii], left)
            min_max_dyn = calculate_min_v_max(root_dyn, func_dyn, choice=choice)

            ax.plot(root_dyn, trial_counter, marker=min_max_dyn['sign'], color = min_max_dyn['color'])
        ax.axhline(trial_counter, linestyle='--', linewidth=1, color='k')
        tick_place.append(trial_counter - trial_counter_participant / 2)
    if dynamic == 'Additive':
        ax.set_yticks(tick_place[1:])
        ax.set_yticklabels(tick_names, minor=False)
    else:
        ax.axes.yaxis.set_visible(False)

    ax.set_xlabel('Riskaversion ($\eta$)')
    ax.set_title(dynamic)

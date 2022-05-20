import numpy as np
import pandas as pd
import os
from scipy import misc

def read_active_data_to_df(root_path:str,
                           dynamics:list[str],
                           subject_numbers:list[int],
                           run:int = 1,
                           choice_dict:dict = {'right': 1, 'left': 0}) -> dict[str,dict[int,pd.DataFrame]]:
    dfs = dict()
    for dynamic in dynamics:
        dynamic_text = {'Additive': '0d0', 'Multiplicative': '1d0'}
        dynamic_dfs = dict()
        for subject in subject_numbers:
            data = pd.read_csv(os.join_path(root_path, f'sub-{subject}_ses-lambd{dynamic_text[dynamic]}_task-active_run-{run}_events'),sep='\t')
            subject_df = data[data['event_type'] == 'Response'].reset_index()
            subject_df['response_button'] = subject_df['response_button'].map(choice_dict)
            dynamic_dfs[subject] = subject_df
        dfs[dynamic] = dynamic_dfs
    return dfs

def read_active_data_to_dict(root_path:str,
                             dynamics:list[str],
                             subject_numbers:list[int],
                             run:int = 1,
                             choice_dict:dict = {'right': 1, 'left': 0}) -> dict[str, any]:
    print('Not implemented yet')
    datadict = dict()
    for dynamic in dynamics:
        dynamic_specs = {'Additive':       {'text': '0d0', 'lambd': 0.0, 'txt_append': '_add'},
                         'Multiplicative': {'text': '1d0', 'lambd': 1.0, 'txt_append': '_mul'}}
        for subject in subject_numbers:
            data = pd.read_csv(os.join_path(root_path, f'sub-{subject}_ses-lambd{dynamic_specs[dynamic]["text"]}_task-active_run-{run}_events'),sep='\t')
            subject_df = data[data['event_type'] == 'Response'].reset_index()
            subject_df['response_button'] = subject_df['response_button'].map(choice_dict)

            '''
            Append which session first (has not been decided yet)
            '''
            if subject > 500:
                datadict.setdefault('MultiplicativeSessionFirst',[]).append(1)
            else:
                datadict.setdefault('MultiplicativeSessionFirst',[]).append(0)

            '''
            Retrieve growth rates and wealth and add to dictionary
            '''
            datadict.setdefault('gr1_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_left_up']))
            datadict.setdefault('gr1_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_left_down']))
            datadict.setdefault('gr2_1'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_right_up']))
            datadict.setdefault('gr2_2'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['gamma_right_down']))

            datadict.setdefault('wealth'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['wealth']))

            '''
            Retrieve keypresses and recode into accept/reject left side gamble
            '''
            datadict.setdefault('choice'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['response_button']))
    return datadict
    #io.savemat(os.path.join(root_path,f'all_data.mat'),datadict,oned_as='row')
    #np.savez(os.path.join(root_path,f'all_data.mat.npz'),datadict = datadict)


def indiference_eta(x1:float, x2:float, x3:float, x4:float, w:float, left:int) -> list:
    if w+x1<0 or w+x2<0 or w+x3<0 or w+x4<0:
        return None, None

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
        return {'color':'orange','sign':'>', 'val': 0} if choice==1 else {'color':'b','sign':'<', 'val': 1}
    else:
        return {'color':'orange','sign':'>', 'val': 0} if choice==0 else {'color':'b','sign':'<', 'val': 1}

def is_statewise_dominated(gamble_pair) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater_equal(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater_equal(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater_equal(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater_equal(min(gamble_pair[1]), min(gamble_pair[0])) )


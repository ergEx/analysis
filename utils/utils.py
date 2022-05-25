import os

import numpy as np
import pandas as pd
from scipy import misc


def isoelastic_utility(x:np.ndarray, eta:float) -> np.ndarray:
    """Isoelastic utility for a given wealth.
    Args:
        x (array):
            Wealth vector.
        eta (float):
            Risk-aversion.
    Returns:
        Vector of utilities corresponding to wealths. For log utility if wealth
        is less or equal to zero, smallest float possible is returned. For other
        utilites if wealth is less or equal to zero, smallest possible utility,
        i.e., specicfic lower bound is returned.
    """
    if eta > 1:
        return ValueError("Not implemented for eta > 1!")

    if np.isscalar(x):
        x = np.asarray((x, ))

    u = np.zeros_like(x, dtype=float)

    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    elif np.isclose(eta, 0): #allow negative values in additive dynamic
        u[x > 0] = (np.power(x[x > 0], 1-eta) - 1) / (1 - eta)
    else:
        bound = (-1) / (1 - eta)
        u[x > 0] = (np.power(x[x > 0], 1-eta) - 1) / (1 - eta)
        u[x <= 0] = bound
    return u

def inverse_isoelastic_utility(u:np.ndarray, eta:float) -> np.ndarray:
    """Inverse isoelastic utility function mapping from utility to wealth.
    Args:
        u (array):
            Utility vector.
        eta (float):
            Risk-aversion.
    Returns:
        Vector of wealths coresponding to utilities.
    """

    if eta > 1:
        return ValueError("Not implemented for eta > 1!")

    if np.isscalar(u):
        u = np.asarray((u, ))

    x = np.zeros_like(u, dtype=float)

    if np.isclose(eta, 1):
        x = np.exp(u)
    elif np.isclose(eta, 0): #allow for negative values in additive dynamic
        x = np.power(u * (1 - eta) + 1, 1 / (1 - eta))
    else:
        bound = (-1) / (1 - eta)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
    return x

def wealth_change(x:np.array, gamma:np.array, lambd:float) -> np.ndarray:
    """Apply isoelastic wealth change.
    Args:
        x (array):
            Initial wealth vector.
        gamma (gamma):
            Growth rates.
        lambd (float):
            Wealth dynamic.
    Returns:
        Vector of updated wealths.
    """

    if np.isscalar(x):
        x = np.asarray((x, ))

    if np.isscalar(gamma):
        gamma = np.asarray((gamma, ))

    return inverse_isoelastic_utility(isoelastic_utility(x, lambd) + gamma, lambd)

def read_active_data_to_df(root_path:str,
                           dynamics:list[str],
                           subject_ids:list[int],
                           run:int = 1,
                           choice_dict:dict = {'right': 1, 'left': 0}) -> tuple(dict[str,dict[int,pd.DataFrame]] , pd.DataFrame):
    dfs = dict()
    full_df = pd.DataFrame()
    for dynamic in dynamics:
        dynamic_text = {'Additive': '0d0', 'Multiplicative': '1d0'}
        dynamic_dfs = dict()
        for subject in subject_ids:
            data = pd.read_csv(os.path.join(root_path, f'sub-{subject}_ses-lambd{dynamic_text[dynamic]}_task-active_run-{run}_events.tsv'),sep='\t')
            subject_df = data[data['event_type'] == 'Response'].reset_index()
            subject_df['response_button'] = subject_df['response_button'].map(choice_dict)
            dynamic_dfs[subject] = subject_df
            full_df = pd.concat([full_df, subject_df])
        dfs[dynamic] = dynamic_dfs
    return dfs, full_df

def read_active_data_to_dict(root_path:str,
                             dynamics:list[str],
                             subject_numbers:list[int],
                             run:int = 1,
                             choice_dict:dict = {'right': 1, 'left': 0}) -> dict[str, any]:
    datadict = dict()
    for dynamic in dynamics:
        dynamic_specs = {'Additive':       {'text': '0d0', 'lambd': 0.0, 'txt_append': '_add'},
                         'Multiplicative': {'text': '1d0', 'lambd': 1.0, 'txt_append': '_mul'}}
        for subject in subject_numbers:
            data = pd.read_csv(os.path.join(root_path, f'sub-{subject}_ses-lambd{dynamic_specs[dynamic]["text"]}_task-active_run-{run}_events.tsv'),sep='\t')
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
            datadict.setdefault('x1_1'+dynamic_specs[dynamic]['txt_append'],[]).append(wealth_change(np.array(subject_df['wealth']), np.array(subject_df['gamma_left_up']), dynamic_specs[dynamic]['lambd']))
            datadict.setdefault('x2_1'+dynamic_specs[dynamic]['txt_append'],[]).append(wealth_change(np.array(subject_df['wealth']), np.array(subject_df['gamma_left_down']), dynamic_specs[dynamic]['lambd']))
            datadict.setdefault('x2_1'+dynamic_specs[dynamic]['txt_append'],[]).append(wealth_change(np.array(subject_df['wealth']), np.array(subject_df['gamma_right_up']), dynamic_specs[dynamic]['lambd']))
            datadict.setdefault('x2_2'+dynamic_specs[dynamic]['txt_append'],[]).append(wealth_change(np.array(subject_df['wealth']), np.array(subject_df['gamma_right_down']), dynamic_specs[dynamic]['lambd']))

            datadict.setdefault('wealth'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['wealth']))

            '''
            Retrieve keypresses and recode into accept/reject left side gamble
            '''
            datadict.setdefault('choice'+dynamic_specs[dynamic]['txt_append'],[]).append(np.array(subject_df['response_button']))
    return datadict
    #io.savemat(os.path.join(root_path,f'all_data.mat'),datadict,oned_as='row')
    #np.savez(os.path.join(root_path,f'all_data.mat.npz'),datadict = datadict)

def indiference_eta(x1:float, x2:float, x3:float, x4:float, w:float, left:int) -> tuple(float, function):
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

def calculate_min_v_max(root:float, func:function, choice:int) -> dict[str, any]:
    dx = misc.derivative(func,root)
    if dx<0:
        return {'color':'orange','sign':'>', 'val': 0} if choice==1 else {'color':'b','sign':'<', 'val': 1}
    else:
        return {'color':'orange','sign':'>', 'val': 0} if choice==0 else {'color':'b','sign':'<', 'val': 1}

def is_statewise_dominated(gamble_pair) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater_equal(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater_equal(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater_equal(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater_equal(min(gamble_pair[1]), min(gamble_pair[0])) )


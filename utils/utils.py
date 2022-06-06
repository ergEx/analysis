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
                           run:int = 1) -> pd.DataFrame:
    full_df = pd.DataFrame()
    for dynamic in dynamics:
        dynamic_text = {'Additive': '0d0', 'Multiplicative': '1d0'}
        for subject in subject_ids:
            data = pd.read_csv(os.path.join(root_path, 'data', f'sub-{subject}_ses-lambd{dynamic_text[dynamic]}_task-active_run-{run}_events.tsv'),sep='\t')
            subject_df = data.query('event_type == "WealthUpdate"').reset_index(drop=True)

            subject_df = add_info_to_df(subject_df, subject)
            full_df = pd.concat([full_df, subject_df])
    return full_df.reset_index(drop=True)

def read_active_data_to_dict(root_path:str,
                             dynamics:list[str],
                             subject_ids:list[int],
                             run:int = 1) -> dict[str, any]:
    dynamic_specs = {'Additive': {'text': '0d0', 'txt_append': '_add'},
                     'Multiplicative': {'text': '1d0', 'txt_append': '_mul'}}
    datadict = dict()
    for dynamic in dynamics:
        for subject in subject_ids:
            data = pd.read_csv(os.path.join(root_path, 'data', f'sub-{subject}_ses-lambd{dynamic_specs[dynamic]["text"]}_task-active_run-{run}_events.tsv'),sep='\t')
            subject_df = data.query('event_type == "WealthUpdate"').reset_index(drop=True)
            subject_df = add_info_to_df(subject_df, subject)

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
    return datadict

def indiference_eta(x1:float, x2:float, x3:float, x4:float, w:float, left:int = -25) -> list[float, any]:
    if x1<0 or x2<0 or x3<0 or x4<0:
        return None, None

    func = lambda x : (((((x1)**(1-x))/(1-x) + ((x2)**(1-x))/(1-x))/2 - ((w)**(1-x))/(1-x))
                    -  ((((x3)**(1-x))/(1-x) + ((x4)**(1-x))/(1-x))/2 - ((w)**(1-x))/(1-x)) )

    ##if we use fsolve:
    #from scipy.optimize import fsolve
    #root_initial_guess = -20
    #root = fsolve(func, root_initial_guess)
    #return root, func

    for i,x in enumerate(np.linspace(left,50,1000)):
        if x == 1:
            continue
        tmp = func(x)
        if i == 0 or np.sign(prev) == np.sign(tmp):
            prev = tmp
        else:
            return x, func

def calculate_min_v_max(root:float, func, choice:int) -> dict[str, any]:
    dx = misc.derivative(func,root)
    if dx<0:
        return {'color':'orange','sign':'>', 'val': 0} if choice==1 else {'color':'b','sign':'<', 'val': 1}
    else:
        return {'color':'orange','sign':'>', 'val': 0} if choice==0 else {'color':'b','sign':'<', 'val': 1}

def is_statewise_dominated(gamble_pair: np.ndarray) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater_equal(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater_equal(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater_equal(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater_equal(min(gamble_pair[1]), min(gamble_pair[0])) )

def add_info_to_df(df:pd.DataFrame, subject:int, choice_dict:dict = {'right': 1, 'left': 0}) -> pd.DataFrame:
    df['selected_side_map'] = df['selected_side'].map(choice_dict)
    df['subject_id'] = [subject]*len(df)

    x1_1 = []
    x1_2 = []
    x2_1 = []
    x2_2 = []
    indif_etas = []
    min_max_sign = []
    min_max_color = []
    min_max_val = []
    for _, ii in enumerate(df.index):
        trial = df.loc[ii, :]
        x_updates = wealth_change(x=trial.wealth,
                                  gamma=[trial.gamma_left_up, trial.gamma_left_down,
                                        trial.gamma_right_up, trial.gamma_right_down],
                                        lambd=trial.eta)
        x1_1.append(x_updates[0] - trial.wealth)
        x1_2.append(x_updates[1] - trial.wealth)
        x2_1.append(x_updates[2] - trial.wealth)
        x2_2.append(x_updates[3] - trial.wealth)

        root, func = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3], trial.wealth)
        if root is not None:
            indif_etas.append(round(root,2))
            tmp = calculate_min_v_max(root, func, trial.selected_side_map)
            min_max_sign.append(tmp['sign'])
            min_max_color.append(tmp['color'])
            min_max_val.append(tmp['val'])
        else:
            indif_etas.append(np.nan)
            min_max_sign.append(np.nan)
            min_max_color.append(np.nan)
            min_max_val.append(np.nan)

    df['x1_1'] = x1_1
    df['x1_2'] = x1_2
    df['x2_1'] = x2_1
    df['x2_2'] = x2_2
    df['indif_eta'] = indif_etas
    df['min_max_sign'] = min_max_sign
    df['min_max_color'] = min_max_color
    df['min_max_val'] = min_max_val

    return df

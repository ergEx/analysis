import os

import numpy as np
import pandas as pd
import statsmodels.api
from scipy import misc
from scipy.optimize import fsolve
from scipy.special import expit, logit
from statsmodels.tools import add_constant


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

def indiference_eta(x1:float, x2:float, x3:float, x4:float) -> list:
    """Calculates indiference_etas for gamble pairs, ie. at which riskaversion is an agent indifferent between the choices
    Args:
        x1 (float): after trial wealth if upper left is realized
        x2 (float): after trial wealth if lower left is realized
        x3 (float): after trial wealth if upper right is realized
        x4 (float): after trial wealth if lower right is realized

    Returns:
        Indifference eta (float).
    """
    if x1<0 or x2<0 or x3<0 or x4<0:
        #print(x1,x2,x3,x4)
        return None, None
        #raise ValueError(f"Isoelastic utility function not defined for negative values")

    func = lambda x : ((isoelastic_utility(x1,x)+ isoelastic_utility(x2,x)) / 2
                     - (isoelastic_utility(x3,x)+ isoelastic_utility(x4,x)) / 2)
    root_initial_guess = -20
    root = fsolve(func, root_initial_guess)

    return root, func

def calculate_min_v_max(root:float, func, choice:int) -> dict[str, any]:
    dx = misc.derivative(func,root)
    if dx<0:
        return np.array([0,0,0]) if choice==1 else np.array([1,1,1])
    else:
        return np.array([0,0,0]) if choice==0 else np.array([1,1,1])

def is_statewise_dominated(gamble_pair: np.ndarray) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater_equal(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater_equal(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater_equal(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater_equal(min(gamble_pair[1]), min(gamble_pair[0])) )

def add_info_to_df(df:pd.DataFrame, subject:str, choice_dict:dict = {'right': 1, 'left': 0}) -> pd.DataFrame:
    df['selected_side_map'] = df['selected_side'].map(choice_dict)
    df['subject_id'] = [subject]*len(df)
    new_info = np.zeros([df.shape[0],8])
    new_info_col_names = ['x1_1','x1_2','x2_1','x2_2','indif_eta','min_max_sign','min_max_color','min_max_val']

    for i, ii in enumerate(df.index):
        trial = df.loc[ii, :]
        x_updates = wealth_change(x=trial.wealth,
                                  gamma=[trial.gamma_left_up, trial.gamma_left_down,
                                        trial.gamma_right_up, trial.gamma_right_down],
                                        lambd=trial.eta)
        new_info[i,0:4] = x_updates - trial.wealth

        root, func = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3])
        if root is not None:
            new_info[i,4] = round(root[0],2)
            new_info[i,5:8] = calculate_min_v_max(root[0], func, trial.selected_side_map)
        else:
            new_info[i,4:8] = np.array([np.nan,np.nan,np.nan,np.nan])

    col_names = list(df.columns) + new_info_col_names
    df = pd.concat([df, pd.DataFrame(new_info)], axis=1)
    df.columns = col_names
    return df

def logistic_regression(df):
    model = statsmodels.api.Logit(np.array(df.min_max_val), add_constant(np.array(df.indif_eta))).fit(disp=0)
    x_test = np.linspace(min(df.indif_eta), max(df.indif_eta), len(df.indif_eta)*5)
    X_test = add_constant(x_test)
    pred = model.predict(X_test)
    se = np.sqrt(np.array([xx@model.cov_params()@xx for xx in X_test]))

    ymin = expit(logit(pred) - 1.96*se)
    ymax = expit(logit(pred) + 1.96*se)

    idx_m = min([i for i in range(len(pred)) if pred[i] > 0.5]) if len([i for i in range(len(pred)) if pred[i] > 0.5]) > 0 else len(x_test) -1
    idx_l = min([i for i in range(len(ymin)) if ymin[i] > 0.5]) if len([i for i in range(len(ymin)) if ymin[i] > 0.5]) > 0 else len(x_test) -1
    idx_h = min([i for i in range(len(ymax)) if ymax[i] > 0.5]) if len([i for i in range(len(ymax)) if ymax[i] > 0.5]) > 0 else len(x_test) -1

    return x_test, pred, ymin, ymax, idx_m, idx_l, idx_h

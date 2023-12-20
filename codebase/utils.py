import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from scipy import misc
from scipy.optimize import fsolve
from statsmodels.tools import add_constant

sns.set_context('paper', font_scale=1.1) #, rc=rcParamsDefault)
cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (6.5 * cm , 5.75 * cm)

plt.rcParams.update({
    "text.usetex": True})

def get_config_filename(argv):
    # Determine the name of the config file to be used
    filename = "config_1_pilot.yaml"
    if len(argv) == 1:
        print("No config file specified. Assuming config_1_pilot.yaml")
    else:
        filename = argv[1]
        print("Using config file ", filename)

    # Check that the config file exists and is readable
    if not os.access(filename, os.R_OK):
        ValueError(f"Config file {filename} does not exist or is not readable. Exiting.")

    return filename

def isoelastic_utility(x: np.ndarray, eta: float) -> np.ndarray:
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
        x = np.asarray((x,))

    u = np.zeros_like(x, dtype=float)

    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    elif np.isclose(eta, 0):  # allow negative values in additive dynamic
        u = (np.power(x, 1 - eta) - 1) / (1 - eta)
    else:
        bound = (-1) / (1 - eta)
        u[x > 0] = (np.power(x[x > 0], 1 - eta) - 1) / (1 - eta)
        u[x <= 0] = bound
    return u


def inverse_isoelastic_utility(u: np.ndarray, eta: float) -> np.ndarray:
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
        raise ValueError("Not implemented for eta > 1!")

    if np.isscalar(u):
        u = np.asarray((u,))

    x = np.zeros_like(u, dtype=float)

    if np.isclose(eta, 1):
        x = np.exp(u)
    elif np.isclose(eta, 0):  # allow for negative values in additive dynamic
        x = np.power(u * (1 - eta) + 1, 1 / (1 - eta))
    else:
        bound = (-1) / (1 - eta)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
    return x


def wealth_change(x: np.array, gamma: np.array, lambd: float) -> np.ndarray:
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
        x = np.asarray((x,))

    if np.isscalar(gamma):
        gamma = np.asarray((gamma,))

    return inverse_isoelastic_utility(isoelastic_utility(x, lambd) + gamma, lambd)


def indiference_eta(x1: float, x2: float, x3: float, x4: float) -> list:
    """Calculates indiference_etas for gamble pairs, ie. at which riskaversion is an agent indifferent between the choices
    Args:
        x1 (float): after trial wealth if upper left is realized
        x2 (float): after trial wealth if lower left is realized
        x3 (float): after trial wealth if upper right is realized
        x4 (float): after trial wealth if lower right is realized

    Returns:
        Indifference eta (float).
    """
    if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
        return None, None
        # raise ValueError(f"Isoelastic utility function not defined for negative values")

    func = lambda x: (
        (isoelastic_utility(x1, x) + isoelastic_utility(x2, x)) / 2
        - (isoelastic_utility(x3, x) + isoelastic_utility(x4, x)) / 2
    )
    root_initial_guess = -20
    root = fsolve(func, root_initial_guess)

    return root, func


def calculate_min_v_max(root: float, func, choice: int) -> np.array:
    """
    Calculate the minimum/maximum values for a given root and a given choice.

    Parameters:
    - root (float): root value.
    - func (function): function to calculate the derivative at root.
    - choice (int): 0 or 1 indicating the choice.

    Returns:
    - np.array indicating 'sign', 'color', and 'val'.
    """
    dx = misc.derivative(func, root)
    if dx < 0:
        return 0 if choice == 0 else 1
    else:
        return 0 if choice == 1 else 1


def is_statewise_dominated(gamble_pair: np.ndarray) -> bool:
    """
    Check if a gamble is strictly statewise dominated by the other gamble in a gamble pair.

    A gamble is strictly statewise dominated if it is worse than another gamble in all possible states.
    In other words, if the minimum and maximum possible outcomes of the gamble are worse than
    the minimum and maximum possible outcomes of the other gamble, respectively, then the gamble
    is strictly statewise dominated.

    Parameters:
    - gamble_pair (np.ndarray): array containing two gambles.

    Returns:
    - bool: True if one of the gambles is strictly statewise dominated, False otherwise.
    """
    # Check if the first gamble is strictly statewise dominated
    if (max(gamble_pair[0]) >= max(gamble_pair[1])) and (
        min(gamble_pair[0]) >= min(gamble_pair[1])
    ):
        return True
    # Check if the second gamble is strictly statewise dominated
    elif (max(gamble_pair[1]) >= max(gamble_pair[0])) and (
        min(gamble_pair[1]) >= min(gamble_pair[0])
    ):
        return True
    # If neither gamble is strictly statewise dominated, return False
    else:
        return False


def logistic_regression(df: pd.DataFrame):
    """Fit a logistic regression model to the data and compute prediction intervals.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to fit the model to.

    Returns:
        tuple: Tuple containing the following elements:
            x_test (np.ndarray): Test data for the predictor variable.
            pred (np.ndarray): Fitted values.
            ymin (np.ndarray): Lower bound of the 95% prediction interval.
            ymax (np.ndarray): Upper bound of the 95% prediction interval.
            idx_m (int): Index of the point where the fitted curve crosses 0.5.
            idx_l (int): Index of the point where the lower bound of the prediction interval crosses 0.5.
            idx_h (int): Index of the point where the upper bound of the prediction interval crosses 0.5.
    """
    model = statsmodels.api.Logit(np.array(df.min_max), add_constant(np.array(df.indif_eta))).fit(
        disp=0
    )

    x_test = np.linspace(-50, 50, 1_000)
    X_test = add_constant(x_test)
    pred = model.predict(X_test)
    cov = model.cov_params()
    gradient = (pred * (1 - pred) * X_test.T).T
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    c = 1.96
    upper = np.maximum(0, np.minimum(1, pred + std_errors * c))
    lower = np.maximum(0, np.minimum(1, pred - std_errors * c))

    idx = np.argmin(np.abs(pred - 0.5))
    slope = np.gradient(pred)[idx]
    decision_boundary = x_test[idx] if slope > 0 else np.nan

    std_dev = (
        (x_test[np.argmin(np.abs(lower - 0.5))] - decision_boundary) / c if slope > 0 else np.nan
    )

    return (
        x_test,
        pred,
        lower,
        upper,
        decision_boundary,
        std_dev,
    )


def read_Bayesian_output(file_path: str) -> dict:
    """Read HLM output file.

    Args:
        Filepath to where the Bayesian output is found

    Returns:
        dict: Dictionary containing the HLM samples.

    """
    mat = mat73.loadmat(file_path)
    return mat["samples"]


def write_provenance(string: str, file: str = 'provenance.txt'):
    """Helper function to write termination and start to provenance.txt
    to help us keep track of what happened.

    Parameters
    ----------
    string : str
        String to print
    file : str, optional
        File to save to, by default 'provenance.txt'
    """
    import datetime
    now=datetime.datetime.now().isoformat()
    out_string = now + '\t' + string + '\n'

    with open(file, 'a+') as f:
        f.write(out_string)

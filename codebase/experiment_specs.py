import itertools


def sub_specs(data_type: str, data_variant: str):
    """
    Returns a dictionary of data specification for the given data variant.

    Parameters
    ----------
    data_variant: str
        The variant of data to retrieve specifications for. Can be one of 'simulation', 'two_gamble_new_c', or 'full_data'.

    Returns a dictionary containing relevant information on the subject structure of the data in the given data variant.
    """
    if data_type == "0_simulation":
        if data_variant == 'full_grid':
            return {
            "id": list(range(260)),
            "first_run": [[1, 2]] * 260,
            }
        elif data_variant == 'varying_variance':
            return {
            "id": list(range(30)),
            "first_run": [[1, 2]] * 30,
            }
        elif data_variant == 'strong_weak_signal':
            return {
            "id": list(range(30)),
            "first_run": [[1, 2]] * 30,
            }
        else:
            return ValueError("Data variant not supported")

    elif data_type == "real_data":
        if data_variant == "1_pilot":
            return {
                "id": [str(i).zfill(3) for i in range(11)],
                "first_run": [
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [2, 1],
                    [2, 1],
                    [2, 1],
                    [1, 2],
                    [2, 1],
                    [2, 1],
                    [2, 1],
                ],
            }
        elif data_variant == "2_full_data":
            raise ValueError("Full data doesn't exist yet")
        else:
            ValueError("Data variant not supported")
    else:
        ValueError("Data type not supported")


def condition_specs():
    """
    Returns a dictionary of data specification for different conditions.

    Returns a ductionary containing relevant information on the condition structure of the data in the given data variant.
    """
    return {
        "condition": ["Additive", "Multiplicative"],
        "lambd": [0.0, 1.0],
        "bids_text": ["0d0", "1d0"],
        "txt_append": ["_add", "_mul"],
        "active_limits": {0.0: [-500, 2_500], 1.0: [64, 15_589]},
    }


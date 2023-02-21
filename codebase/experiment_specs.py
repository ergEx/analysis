import itertools


def sub_specs(data_variant: str) -> dict[str, list]:
    """
    Returns a dictionary of data specification for the given data variant.

    Parameters
    ----------
    data_variant: str
        The variant of data to retrieve specifications for. Can be one of 'simulation', 'two_gamble_new_c', or 'full_data'.

    Returns a dictionary containing relevant information on the subject structure of the data in the given data variant.
    """
    if data_variant == "0_simulation":
        etas = [-0.5, 0.0, 0.5, 1.0, 1.5]
        return {
            "id": [f"{i[0]}x{i[1]}" for i in itertools.product(etas, etas)],
            "first_run": [[1, 2]] * len(etas) ** 2,
        }
    elif data_variant == "1_pilot":
        return {
            # ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
        raise ValueError("Unknown variant")


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


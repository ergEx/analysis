import os


def sub_specs(data_type: str, data_variant: str, in_folder: str = None):
    """
    Returns a dictionary of data specification for the given data variant.

    Parameters
    ----------
    data_variant: str
        The variant of data to retrieve specifications for. Can be one of 'simulation', 'two_gamble_new_c', or 'full_data'.

    Returns a dictionary containing relevant information on the subject structure of the data in the given data variant.
    """

    if in_folder is None:
        in_folder = os.path.join('data', data_variant)

    if data_type == "0_simulation":
        if data_variant == 'grid':
            return {
            "id": list(range(10)),
            "first_run": [[1, 2]] * 10,
            }
        elif data_variant == 'varying_variance':
            return {
            "id": list(range(20)),
            "first_run": [[1, 2]] * 20,
            }
        elif data_variant == 'strong_weak_signal':
            return {
            "id": list(range(20)),
            "first_run": [[1, 2]] * 20,
            }
        else:
            return ValueError("Data variant not supported")

    elif data_type == "real_data":

        return create_spec_dict(in_folder)

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


def create_spec_dict(folder):
    from glob import glob
    import re

    subs = sorted([i.split('/')[-1].split('-')[-1] for i in glob(f'{folder}/sub-*')])
    order = []

    pattern = r"acq-(\w+)_run"

    for ii in subs:
        file = glob(f'{folder}/sub-{ii}/ses-1/*passive*_run-1*')[0]
        match = re.search(pattern, file)
        if match:
            extracted_part = match.group(1)

            if extracted_part == 'lambd0d0':
                order.append([1, 2])
            elif extracted_part == 'lambd1d0':
                order.append([2 ,1])
            else:
                raise ValueError(f"Something went wrong with {ii}")

        else:
            raise ValueError("No match?!")

    return {'id': subs, 'first_run': order}

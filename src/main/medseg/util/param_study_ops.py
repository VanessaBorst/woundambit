# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

def encode_trial_name_short(trial_params, separator="_"):
    """
    Encode all values from the trial_params dict into a string for a trial name.

    Args:
        trial_params (dict): Dictionary of trial-specific parameters.
        separator (str): Separator between values.

    Returns:
        str: Encoded string representing the trial name.
    """

    def extract_values(d):
        """
        Recursively extract values from a nested dictionary in order.

        Args:
            d (dict): Dictionary to extract values from.

        Returns:
            list: List of values in the order they appear.
        """
        values = []
        for v in d.values():
            if isinstance(v, dict):
                values.extend(extract_values(v))
            else:
                values.append(v)
        return values

    # Extract all values while preserving order
    values = extract_values(trial_params)

    # Convert to a string using the provided separator
    return separator.join(map(str, values))

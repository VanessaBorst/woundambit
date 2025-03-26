def nested_get(dictionary, keys, default=None):
    """
    Perform a nested lookup in a dictionary with a list of keys.

    Args:
        dictionary (dict): The dictionary to perform the lookup in.
        keys (list): List of keys representing the path to the desired value.
        default: The default value to return if any key is missing.

    Returns:
        The value at the nested key path, or the default value.
    """
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary


def integer_float_to_int(value):
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Cannot convert {value} to int because it is not a whole number.")
    return int(value)
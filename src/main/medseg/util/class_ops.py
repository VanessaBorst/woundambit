import builtins
import inspect
from typing import Any, Dict, Set


def get_class(obj: object, class_name: str):
    """
    Get a class from an object by name.
    """
    for a in dir(obj):
        if a == class_name or a.lower() == class_name.lower() or a.lower() == class_name.lower().replace("_", ""):
            return builtins.getattr(obj, a)
    return None


def get_class_arg_names(cls: Any) -> Set[str]:
    """
    Extracts the argument names from the __init__ method of a given class.

    :param cls: The class whose __init__ method's argument names we want to extract.
    :return: A set containing the argument names.
    """
    init_signature = inspect.signature(cls.__init__)
    # Exclude the 'self' parameter, as it is the instance itself
    arg_names = {param.name for param in init_signature.parameters.values() if param.name != 'self'}
    return arg_names


def get_init_args(instance: Any) -> Dict[str, Any]:
    """
    Extracts the arguments and their values from the __init__ method of a given class instance,
    assuming the arguments have been set as class attributes with the same names.

    :param instance: An instance of the class whose __init__ method's arguments we want to extract.
    :return: A dictionary containing the argument names as keys and their values as values.
    """
    init_arg_names = get_class_arg_names(instance.__class__)
    init_args = {arg_name: getattr(instance, arg_name) for arg_name in init_arg_names if hasattr(instance, arg_name)}
    return init_args


def check_if_method_defined(instance: Any, method_name: str) -> bool:
    """
    Checks if a method is defined in a class instance.

    :param instance: The instance of the class to check.
    :param method_name: The name of the method to check.
    :return: True if the method is defined, False otherwise.
    """
    return hasattr(instance, method_name) and callable(getattr(instance, method_name))

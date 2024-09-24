import json
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))


def get_hyperparameters(hyperparam_filename):
    """
    Get the hyperparameters from a JSON file.

    Args:
        hyperparam_filename (str): The name of the JSON file (e.g. '0.json')
    Returns:
        dict: The hyperparameters as a dictionary.
    """
    with open(f'{HYPERPARAM_DIR}/{hyperparam_filename}', 'r') as f:
        hyperparams = json.load(f)
    return hyperparams
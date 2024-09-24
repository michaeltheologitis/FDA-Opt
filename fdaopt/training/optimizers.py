from fdaopt.utils import torch


def get_opt_hyperparams(where, hyperparams):
    """
    Extracts optimizer hyperparameters from the given hyperparameters dictionary.

    Args:
        where (str): A prefix indicating whether the optimizer is 'server' or 'client'.
        hyperparams (dict): A dictionary containing hyperparameters including optimizer hyperparameters. It is
            expected it contains `server_opt_name` and `client_opt_name`. The rest of the optimizer hyperparameters
            are expected to have keys `server_opt_HYPERPARAM` or `client_opt_HYPERPARAM` where the HYPERPARAM is
            a valid optimizer hyperparameter (for example, `lr`).

    Returns:
        dict: A dictionary containing the extracted optimizer hyperparameters.
    """

    opt_dict = {}
    where_opt = f"{where}_opt"

    for k, v in hyperparams.items():
        if where_opt in k:
            param = k.replace(f"{where_opt}_", '')
            # Skip the 'name' key specifying the optimizer
            if param != 'name':
                opt_dict[param] = v

    return opt_dict


def get_optimizer_class(optimizer_name):
    """
    Dynamically imports and returns the optimizer class from torch.optim based on the given optimizer name.

    Args:
        optimizer_name (str): The name of the optimizer class as a string.

    Returns:
        type: The optimizer class from torch.optim.
    """
    optimizer_class = getattr(torch.optim, optimizer_name)
    return optimizer_class


def server_client_optimizers(train_params, hyperparams):
    """
    Creates and returns the server and client optimizers based on the provided hyperparameters.

    Args:
        train_params (list of torch.nn.Parameter): The parameters of the model to be optimized.
        hyperparams (dict): A dictionary containing hyperparameters including optimizer hyperparameters. It is
            expected it contains `server_opt_name` and `client_opt_name`. The rest of the optimizer hyperparameters
            are expected to have keys `server_opt_HYPERPARAM` or `client_opt_HYPERPARAM` where the HYPERPARAM is
            a valid optimizer hyperparameter (for example, `lr`).

    Returns:
        tuple: A tuple containing the server optimizer and the client optimizer.
    """

    server_opt_class = get_optimizer_class(hyperparams['server_opt_name'])
    client_opt_class = get_optimizer_class(hyperparams['client_opt_name'])

    server_opt = server_opt_class(train_params, **get_opt_hyperparams('server', hyperparams))
    client_opt = client_opt_class(train_params, **get_opt_hyperparams('client', hyperparams))

    return server_opt, client_opt

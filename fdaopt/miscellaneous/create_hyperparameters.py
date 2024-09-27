import json
import os
import argparse
import ast

script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))

def get_opt_args(opt_args, where):
    """ Get the optional arguments for the optimizer.
    Args:
        opt_args (list): The list of optional arguments.
        where (str): Either "server_opt" or "client_opt".
    Returns:
        dict: The optional arguments as a dictionary.
    """
    opt_params = {}

    if opt_args:
        for arg in opt_args:
            key, value = arg.split('=')
            opt_params[f"{where}_{key}"] = ast.literal_eval(value)

    return opt_params

def get_next_filename():
    """ Get the next filename for the hyperparameters.
    Returns:
        str: The next filename for the hyperparameters.
    """
    files = os.listdir(HYPERPARAM_DIR)
    return f"{len(files)}"

def create_hyperparameters(args):
    hyperparams = {
        'checkpoint': args.checkpoint,
        'ds_path': args.ds_path,
        'ds_name': args.ds_name,
        'num_labels': args.num_labels,
        'num_clients': args.num_clients,
        'clients_per_round': args.clients_per_round,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'local_epochs': args.local_epochs,
        'total_rounds': args.total_rounds,
        'fda': args.fda,
        'theta': args.theta,
        'extras': args.extras
    }

    server_opt_args = get_opt_args(args.server_opt_args, "server_opt")
    client_opt_args = get_opt_args(args.client_opt_args, "client_opt")

    hyperparams = {
        **hyperparams,
        **{'server_opt_name': args.server_opt_name},
        **server_opt_args,
        **{'client_opt_name': args.client_opt_name},
        **client_opt_args
    }

    filename = get_next_filename()

    with open(f'{HYPERPARAM_DIR}/{filename}.json', 'w') as f:
        json.dump(hyperparams, f)

    print(f"OK! Created hyperparameters in {filename}.json.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, help="The checkpoint to use for the model.", default='roberta-base')
    parser.add_argument('--ds_path', type=str, help="The path to the dataset.", default='glue')
    parser.add_argument('--ds_name', type=str, help="The name of the dataset.", default='mrpc')
    parser.add_argument('--num_labels', type=int, help="The number of labels in the dataset.", default=2)
    parser.add_argument('--num_clients', type=int, help="The number of clients in the federated learning setup.", default=100)
    parser.add_argument('--clients_per_round', type=int, help="The number of clients to sample per round.", default=10)
    parser.add_argument('--alpha', type=float, help="The alpha parameter of the Dirichlet distribution.", default=1.)
    parser.add_argument('--batch_size', type=int, help="The batch size for the training.", default=8)
    parser.add_argument('--local_epochs', type=int, help="The number of local epochs until we synchronize the model.", default=1)
    parser.add_argument('--total_rounds', type=int, help="The total number of rounds for the training.", default=500)
    parser.add_argument('--fda', action='store_true', help="If given, then we train with FDA.")
    parser.add_argument('--theta', type=float, help="The theta parameter for the FDA.", default=0.0)
    parser.add_argument('--server_opt_name', type=str, help="The name of the server optimizer.", required=True)
    parser.add_argument('--client_opt_name', type=str, help="The name of the client optimizer.", required=True)

    parser.add_argument('--server_opt_args', nargs='*', help="Hyperparameters for the server optimizer (e.g., --server_opt_args lr=0.001 betas='(0.9, 0.999)'")
    parser.add_argument('--client_opt_args', nargs='*', help="Hyperparameters for the client optimizer (e.g., --client_opt_args lr=0.001")

    parser.add_argument('--extras', type=str, help="Extra info.", default='')

    create_hyperparameters(parser.parse_args())

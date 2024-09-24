from fdaopt import utils

utils.set_seed(42)
utils.set_device('cuda:1')

if __name__ == '__main__':
    from fdaopt.training.fed_opt import fed_opt
    
    hyperparams = {
        'checkpoint': 'roberta-base',
        'ds_path': 'glue',
        'ds_name': 'mrpc',
        'num_labels': 2,
        'num_clients': 100,
        'clients_per_round': 10,
        'alpha': 1.,
        'batch_size': 8,
        'local_epochs': 1,
        'total_rounds': 1000,
        'server_opt_name': 'Adam',
        'server_opt_lr': 0.0001,
        'client_opt_name': 'SGD',
        'client_opt_lr': 0.001
    }

    fed_opt(hyperparams)

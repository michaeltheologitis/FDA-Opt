

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="The filename of the hyperparameters.")
    parser.add_argument('--device', type=str, help="The device to use (e.g. 'cuda:1').")
    parser.add_argument('--seed', type=int, default=42, help="The seed for reproducibility.")
    args = parser.parse_args()

    from fdaopt import utils

    # Set the seed and the device before importing the rest of the modules
    utils.set_seed(args.seed)
    utils.set_device(args.device)

    from fdaopt.training.fed_opt import fed_opt
    from fdaopt.miscellaneous.parameter_tools import get_hyperparameters

    hyperparams = get_hyperparameters(args.filename)

    print(f"Using device: {utils.DEVICE}")
    print(hyperparams)

    fed_opt(hyperparams)

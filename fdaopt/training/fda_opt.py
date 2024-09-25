import gc

from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
    update_sampled_client_parameters, variance, compute_metrics, fda_ksi_vector, fda_variance_approx)
from fdaopt.training.fed_train import federated_training_step
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, np


def estimation_of_theta(round_variances, p=1/3):
    """
    Estimate the weighted average of the provided round variances using a custom weighting scheme.

    This function computes the weighted average (or weighted sum) of a series of variance values (from different rounds
    or samples) using a non-uniform weighting strategy. The weight for each round variance increases with its index in
    the sequence, giving more importance to later round variances. The weighting strategy is based on a power-law,
    with a default exponent of 1/3, meaning weights increase sub-linearly with the index.

    Args:
        round_variances (list): A sequence of variance values (e.g., from different rounds) for which the weighted sum
                                will be computed.
        p (float, optional): The exponent controlling how sharply weights increase with the index. Default is 1/3,
                             which applies sub-linear weights. Increase this value to emphasize later variances more.
    Returns:
        float: The weighted average (linear estimation) of the provided round variances.
    """

    def weights_for_weighted_sum(_p, _n):
        """
        Compute custom weights w_i = (i^p) / sum(i^p) for i from 1 to n.

        Args:
            _p (float): The exponent controlling how sharply weights increase with the index.
            _n (int): The number of variance values (rounds) being weighted.

        Returns:
            np.array: Normalized weights where each weight corresponds to a variance,
                      and the sum of the weights equals 1.
        """
        _indices = np.arange(1, _n + 1)
        _weights = _indices ** _p
        _normalized_weights = _weights / np.sum(_weights)

        return _normalized_weights

    # number of samplings
    n = len(round_variances)

    weights = weights_for_weighted_sum(p, n)

    # calculate the linear-weighted sum
    weighted_sum = np.dot(weights, round_variances)

    return weighted_sum


def fda_opt(hyperparams):

    fed_ds, test_ds = prepare_federated_datasets(
        hyperparams['ds_path'],
        hyperparams['ds_name'],
        hyperparams['checkpoint'],
        hyperparams['num_clients'],
        hyperparams['alpha'],
        hyperparams['batch_size']
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['checkpoint'],
        num_labels=hyperparams['num_labels']
    )

    # Move the model to the `DEVICE`
    model = model.to(DEVICE)

    # Extract trainable parameters from the model, which reside on the device that the model resides in
    train_params = [param for param in model.parameters() if param.requires_grad]

    # Create a copy of the trainable parameters in `DEVICE`, detached from the computation graph
    round_start_train_params = [param.detach().clone() for param in train_params]

    server_opt, client_opt = server_client_optimizers(train_params, hyperparams)

    client_sampler = ClientSampler(
        list(range(hyperparams['num_clients'])),
        hyperparams['clients_per_round']
    )

    client_train_params = {
        client_id: [param.detach().clone() for param in round_start_train_params]
        for client_id in range(hyperparams['clients_per_round'])
    }

    metrics_handler = MetricsHandler(hyperparams)

    # 1. ESTIMATE THETA THRESHOLD. RUN FedOpt FOR A FEW ROUNDS TO ESTIMATE THE THETA THRESHOLD

    # Number of rounds to estimate the theta threshold
    theta_estimation_rounds = int(hyperparams['num_clients'] / hyperparams['clients_per_round'])

    # List to store the variances of the rounds for theta estimation
    round_variances = []

    ksi = None

    for r in range(theta_estimation_rounds):

        training_loss = 0.0

        # Save the model parameters at the start of this round
        sampled_clients = client_sampler.sample()

        # Save the model parameters at the start of this round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        update_sampled_client_parameters(client_train_params, sampled_clients, round_start_train_params)

        # Calculate the total number of steps for this epoch/round
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

        for step in range(round_steps):
            # Perform a federated training step and accumulate the training loss
            training_loss += federated_training_step(model, train_params, client_train_params, client_opt, fed_ds)

        # Reset the model parameters to the parameters at the start of the round. This ensures that the
        # server-side optimizer updates are applied correctly (on the parameters at the start of the round)
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )

        # Compute the drifts (differences) between the round start parameters and the client parameters
        client_drifts = compute_client_drifts(round_start_train_params, client_train_params)

        # Compute pseudo-gradients based on the average drifts
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        # Set the computed pseudo-gradients to the trainable parameters
        set_gradients(train_params, pseudo_gradients)

        # Update model parameters (global model) using the server optimizer based on pseudo-gradients
        server_opt.step()

        # Zero the gradients before the next backward pass
        server_opt.zero_grad()

        # Calculate the ksi vector for the later LinearFDA. The model after the most recent sync is `train_params`,
        # and the model after the 2nd most recent sync is `round_start_train_params`, at this moment.
        ksi = fda_ksi_vector(train_params, round_start_train_params)

        # Calculate evaluation metrics on the test set
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        # Calculate the average training loss for the round
        metrics['training_loss'] = training_loss / round_steps
        # Calculate variance and helpful metrics
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        # Add epoch steps to metrics
        metrics['round_steps'] = round_steps
        # Pass round metrics to handler
        metrics_handler.append_round_metrics(metrics)

        # Append the variance of the round to the list
        round_variances.append(metrics['variance'])

        print(metrics)

        gc.collect()

    # 2. RUN THE FDA-Opt ALGORITHM USING THE ESTIMATED THETA THRESHOLD

    # The variance threshold estimated using the linear-weighted-sum of the round variances
    theta = estimation_of_theta(round_variances)

    for r in range(r + 1, hyperparams['total_rounds'] - theta_estimation_rounds + r + 1):

        training_loss = 0.0

        # Save the model parameters at the start of this round
        sampled_clients = client_sampler.sample()

        # Save the model parameters at the start of this round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        update_sampled_client_parameters(client_train_params, sampled_clients, round_start_train_params)

        round_steps = 0
        var_approx = 0.0

        while var_approx <= theta:
            # Perform a federated training step and accumulate the training loss
            training_loss += federated_training_step(model, train_params, client_train_params, client_opt, fed_ds)

            # Compute the drifts (differences) between the round start parameters and the client parameters
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params)
            # Calculate the current variance approximation with LinearFDA
            var_approx = fda_variance_approx(client_drifts, ksi)

            round_steps += 1

            print(f"Round Steps: {round_steps}, Variance Approximation: {var_approx}, Theta: {theta}")

        # Reset the model parameters to the parameters at the start of the round. This ensures that the
        # server-side optimizer updates are applied correctly (on the parameters at the start of the round)
        copy_parameters(
            from_params=round_start_train_params,
            to_params=train_params
        )

        # Compute pseudo-gradients based on the average drifts
        pseudo_gradients = compute_pseudo_gradients(client_drifts)

        # Set the computed pseudo-gradients to the trainable parameters
        set_gradients(train_params, pseudo_gradients)

        # Update model parameters (global model) using the server optimizer based on pseudo-gradients
        server_opt.step()

        # Zero the gradients before the next backward pass
        server_opt.zero_grad()

        # Calculate the ksi vector for the next round for LinearFDA. The model after the most recent sync is
        # `train_params`, and the model after the 2nd most recent sync is `round_start_train_params`, at this moment.
        ksi = fda_ksi_vector(train_params, round_start_train_params)

        # Calculate evaluation metrics on the test set
        metrics = {"round": r + 1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
        # Calculate the average training loss for the round
        metrics['training_loss'] = training_loss / round_steps
        # Calculate variance and helpful metrics
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        # Add epoch steps to metrics
        metrics['round_steps'] = round_steps
        # Pass round metrics to handler
        metrics_handler.append_round_metrics(metrics)

        # Append the variance of the round to the list
        round_variances.append(metrics['variance'])

        print(metrics)

        gc.collect()



    metrics_handler.save_metrics()
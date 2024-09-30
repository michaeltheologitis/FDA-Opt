import gc

from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
    update_sampled_client_parameters, variance, compute_metrics, fda_variance_approx)
from fdaopt.training.fed_train import federated_training_step
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, np, random


def random_synchronize(e, k=0.6, e_0=9):
    """
    Compute whether to synchronize models at iteration e using a probabilistic approach.
    The probability of synchronization is modeled using a sigmoid function, which smoothly
    increases as e increases. For small values of e~<(e_0 - 1), the probability remains very low,
    and it grows to nearly 1 as e approaches larger values e>(e_0+2).

    Args:
        e (int or float): The current epoch or iteration number. As e increases, the probability
                          of synchronization increases.
        k (float): The steepness of the sigmoid curve. A larger value of k causes the probability
                   to increase faster. Default is 0.6.
        e_0 (int or float): The midpoint of the sigmoid curve where the probability of synchronization
                            is approximately 0.5. Default is 9.

    Returns:
        bool: A boolean indicating whether to synchronize models based on the computed probability.
              Returns False if e equals 0, regardless of the sigmoid function.
    """

    def _sigmoid(_e, _k, _e_0):
        return 1 / (1 + np.exp(-_k * (e - _e_0)))

    if e == 0:
        return False

    # Calculate the probability using the sigmoid function
    p_sync = _sigmoid(e, k, e_0)

    # Return True if we synchronize, False otherwise (based on p_sync)
    return p_sync > random.random()


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

    # Initialize the AMS sketch
    ams_sketch = AmsSketch()

    metrics_handler = MetricsHandler(hyperparams)

    for r in range(hyperparams['total_rounds']):

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
        var_approx = 0.0
        local_epochs = 0
        while var_approx <= hyperparams['theta'] and not random_synchronize(local_epochs):

            local_epochs += 1

            for step in range(round_steps):
                # Perform a federated training step and accumulate the training loss
                training_loss += federated_training_step(model, train_params, client_train_params, client_opt, fed_ds)

            # Compute the drifts (differences) between the round start parameters and the client parameters
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params)
            # Calculate the current variance approximation with LinearFDA
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)

        # TODO: Logically change the names cuz it is not consistent
        round_steps = local_epochs * round_steps

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

        print(metrics)

        gc.collect()



    metrics_handler.save_metrics()
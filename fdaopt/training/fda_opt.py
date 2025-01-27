import gc

from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, ClientSampler
from fdaopt.metrics.mentrics_handler import MetricsHandler
from fdaopt.models.ops import (compute_client_drifts, compute_pseudo_gradients, set_gradients, copy_parameters,
    get_updated_client_parameters, variance, compute_metrics, fda_variance_approx)
from fdaopt.training.fed_train import client_round_train
from fdaopt.training.sketch import AmsSketch
from fdaopt.training.optimizers import server_client_optimizers
from fdaopt.utils import DEVICE, AutoModelForSequenceClassification, np, random, DEVICE_RAM_PROGRAM

# Let RAM usage of one model be M, then,
SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance' or DEVICE_RAM_PROGRAM == 'moderate':
    # All client models are stored in DEVICE, which means DEVICE-RAM is O(num_clients * M)
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'low':
    # All client models are stored in CPU, which means DEVICE-RAM is O(M)
    SAVE_DEVICE = 'cpu'


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

    # A dictionary where keys are client IDs and values are lists of parameter tensors all lying on the SAVE_DEVICE
    client_train_params_dict = {
        client_id: [param.detach().clone().to(SAVE_DEVICE) for param in round_start_train_params]
        for client_id in range(hyperparams['clients_per_round'])
    }

    # Initialize the AMS sketch
    ams_sketch = AmsSketch()

    metrics_handler = MetricsHandler(hyperparams)

    theta = hyperparams['theta']
    first_round = True

    for r in range(hyperparams['total_rounds']):

        training_loss = 0.0

        # Save the model parameters at the start of this round
        sampled_clients = client_sampler.sample()

        # Save the model parameters at the start of this round
        copy_parameters(
            from_params=train_params,
            to_params=round_start_train_params
        )

        client_train_params_dict = get_updated_client_parameters(client_train_params_dict, sampled_clients, round_start_train_params)

        # Calculate the total number of steps for this epoch/round
        round_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)
        var_approx = 0.0
        round_epochs = 0

        #while var_approx <= hyperparams['theta'] and not random_synchronize(round_epochs):
        while var_approx <= theta and round_epochs < hyperparams['max_epochs']:

            round_epochs += 1

            for client_id in sampled_clients:
                training_loss += client_round_train(model, train_params, client_train_params_dict, client_opt, fed_ds,
                                                    client_id, round_steps)

            # Compute the drifts (differences) between the round start parameters and the client parameters
            client_drifts = compute_client_drifts(round_start_train_params, client_train_params_dict)
            # Calculate the current variance approximation with LinearFDA
            var_approx = fda_variance_approx(client_drifts, ams_sketch=ams_sketch)

            if first_round:
                first_round = False
                break

        # Calculate the accumulate training loss for the round for all clients (which might contain multiple epochs)
        training_loss = training_loss / round_epochs

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
        metrics['training_loss'] = training_loss / hyperparams['clients_per_round']
        # Calculate variance and helpful metrics
        metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
        # Add epoch steps to metrics
        metrics['round_steps'] = round_epochs * round_steps
        # Pass round metrics to handler
        metrics_handler.append_round_metrics(metrics)

        print(metrics)

        theta = ((hyperparams['max_epochs'] / 2) / round_epochs) * metrics['variance']


        # TODO: Change rigid logic
        if hyperparams['ds_name'] == 'mrpc':
            if metrics['accuracy'] >= 0.902 * 0.95:
                break
        if hyperparams['ds_name'] == 'sst-2':
            if metrics['accuracy'] >=  0.948 * 0.99:
                break

        gc.collect()

    metrics_handler.save_metrics()
from fdaopt.utils import torch, evaluate, DEVICE

@torch.no_grad
def copy_parameters(from_params, to_params):
    """
    Copies the values from one set of parameters to another.

    This function operates in-place and modifies the `to_parameters` directly.
    The @torch.no_grad() decorator ensures that this operation is not tracked
    by autograd, preventing unnecessary computation and memory usage.

    Args:
        from_params (list of torch.nn.Parameter): An iterable of source parameters to copy from.
        to_params (list of torch.nn.Parameter): An iterable of destination parameters to copy to.
    """
    for from_param, to_param in zip(from_params, to_params):
        to_param.copy_(from_param)


@torch.no_grad
def average_client_parameters(client_train_params):
    """
    Averages the parameters from multiple clients.

    This function computes the mean of the parameters from all clients. It stacks
    the parameters for each layer across clients, computes the mean, and returns
    the averaged parameters. The @torch.no_grad decorator ensures that this operation
    is not tracked by autograd.

    Args:
        client_train_params (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.

    Returns:
        list: A list of averaged parameters.
    """
    average_params = [
        torch.mean(torch.stack(param_list), dim=0)
        for param_list in zip(*client_train_params.values())
    ]

    return average_params


@torch.no_grad
def compute_metrics(model, ds_path, ds_name, test_ds):
    """
    Computes evaluation metrics for the given model on the test dataset.

    This function evaluates the model on the provided test dataset and computes
    metrics using the `evaluate` library. The `@torch.no_grad` decorator ensures
    that the evaluation is performed without tracking gradients, which saves memory
    and computation.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        ds_path (str): The path or identifier of the dataset. This is used to load the appropriate evaluation metric.
        ds_name (str): The name of the specific dataset configuration. This helps in loading the correct evaluation metric.
        test_ds (DataLoader): A DataLoader for the test dataset.

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """

    # Load the evaluation metric
    metric = evaluate.load(path=ds_path, config_name=ds_name)

    testing_loss = 0.0
    num_batches = len(test_ds)

    # Set the model to evaluation mode
    model.eval()

    for batch in test_ds:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Perform a forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Get logits and predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Add batch predictions and references to the metric
        metric.add_batch(predictions=predictions, references=batch["labels"])

        testing_loss += loss.item()

    # Calculate the average test loss
    average_test_loss = testing_loss / num_batches

    # Compute the final evaluation metrics
    metrics = metric.compute()

    # Add the average test loss to the evaluation metrics
    metrics['testing_loss'] = average_test_loss

    # Compute and return the final evaluation metrics
    return metrics


@torch.no_grad
def compute_drifts(old_params, new_params):
    """
    Calculate the drift (difference) between old and new parameters.

    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        new_params (list of torch.nn.Parameter): The updated parameters.

    Returns:
        list of torch.Tensor: The computed drifts for each parameter.
    """

    return [
        new_param - old_param
        for old_param, new_param in zip(old_params, new_params)
    ]


@torch.no_grad
def compute_client_drifts(old_params, client_train_params):
    """
    Compute the drifts for all clients based on the original parameters.

    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        client_train_params (dict): Dictionary of client IDs and their corresponding parameters.

    Returns:
        dict: A dictionary where keys are client IDs and values are lists of drifts for each parameter.
    """
    return {
        client_id: compute_drifts(old_params, client_params)
        for client_id, client_params in client_train_params.items()
    }


@torch.no_grad
def compute_pseudo_gradients(client_drifts):
    """
    Compute the pseudo-gradient based on the drifts between old and client parameters.

    Args:
        client_drifts (dict): Dictionary of client IDs and their corresponding parameter drifts.

    Returns:
        list of torch.Tensor: The computed pseudo-gradient.
    """
    average_drifts = average_client_parameters(client_drifts)

    pseudo_gradients = [-drift for drift in average_drifts]

    return pseudo_gradients


@torch.no_grad
def set_gradients(train_params, gradients):
    """
    Set gradients for trainable parameters.

    This function assigns the provided gradients to the .grad attribute of the corresponding trainable parameters.

    Args:
        train_params (list of torch.nn.Parameter): The trainable parameters of the model.
        gradients (list of torch.Tensor): The gradients to be assigned to the parameters.
    """
    for param, gradient in zip(train_params, gradients):
        param.grad = gradient


@torch.no_grad
def vectorize(parameters):
    """
    Concatenates a list of parameter tensors into a single vector.

    Args:
        parameters (list of torch.nn.Parameter): An iterable of parameter tensors.

    Returns:
        torch.Tensor: A single vector containing all the elements of the input parameters.
    """
    return torch.cat([param.view(-1) for param in parameters])


@torch.no_grad
def variance(client_drifts):
    """
    Computes the variance of the client models utilizing the drifts (see paper)

    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).

    Returns:
        float: The computed variance of the client drifts.
    """

    # Vectorize each client's drifts
    drifts_vecs = [vectorize(drifts) for drifts in client_drifts.values()]
    # Compute the squared l2 norms of each client's drifts
    norm_sq_drifts = [torch.dot(vec, vec) for vec in drifts_vecs]
    # Compute the average of the squared norms of the individual client drifts
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)

    # Compute the average drift
    avg_drift = average_client_parameters(client_drifts)
    # Vectorize the average drift
    avg_drift_vec = vectorize(avg_drift)
    # Compute the squared l2 norm of the average drift
    norm_sq_avg_drift = torch.dot(avg_drift_vec, avg_drift_vec)

    # variance of the client models
    var = avg_norm_sq_drifts - norm_sq_avg_drift

    return var.item(), avg_norm_sq_drifts.item(), norm_sq_avg_drift.item()


@torch.no_grad
def update_sampled_client_parameters(client_params, sampled_clients, params):
    """
    This function updates the parameters of the sampled clients with the current global parameters.
    It reassigns the entries in the client_train_params dictionary based on the new sampled clients.

    Args:
        client_params (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        sampled_clients (list): A list of newly sampled client IDs.
        params (list of torch.nn.Parameter): The current global parameters to be assigned to the sampled clients.
    """
    # List of current client IDs
    old_clients = list(client_params.keys())

    # Iterate over pairs of new client IDs and old client IDs
    for new_client_id, old_client_id in zip(sampled_clients, old_clients):
        # Reassign the entry in client_train_params dictionary
        client_params[new_client_id] = client_params.pop(old_client_id)

        # Copy the global parameters to the new client's parameters
        copy_parameters(
            from_params=params,
            to_params=client_params[new_client_id]
        )

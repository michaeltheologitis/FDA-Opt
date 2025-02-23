from fdaopt.utils import torch, evaluate, DEVICE, DEVICE_RAM_PROGRAM

# Let all clients models utilize C amount of RAM, then,
SAVE_DEVICE = None
if DEVICE_RAM_PROGRAM == 'performance':
    # This means we use DEVICE-RAM O(2 * C) -- Essentially, all drifts remain on GPU
    SAVE_DEVICE = DEVICE
elif DEVICE_RAM_PROGRAM == 'moderate' or DEVICE_RAM_PROGRAM == 'low':
    # This means we use DEVICE-RAM O(C) -- Essentially, intermidiate drifts remain on CPU
    SAVE_DEVICE = 'cpu'

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
        device = to_param.device
        to_param.copy_(from_param.to(device))


@torch.no_grad
def average_client_parameters(client_train_params_dict):
    """
    Efficiently averages parameters from multiple clients on GPU.

    This function uses an incremental mean calculation to avoid creating large
    intermediate tensors, performing in-place additions to keep memory usage low.

    Args:
        client_train_params_dict (dict): Dictionary where keys are client IDs and values are lists of parameter tensors.

    Returns:
        list: A list of averaged parameters, stored on `DEVICE`.
    """

    num_clients = len(client_train_params_dict)
    average_params = None

    for train_params in client_train_params_dict.values():

        if not average_params:
            # Initialize with the first client's parameters divided by num_clients
            average_params = [param.clone().to(DEVICE) / num_clients for param in train_params]
        else:
            # Incrementally add other clients' parameters
            for param, avg_param in zip(train_params, average_params):
                avg_param.add_(param.to(DEVICE) / num_clients)

    return average_params

@torch.no_grad
def average_client_parameters2(client_train_params_dict):
    """
    Averages the parameters from multiple clients.

    This function computes the mean of the parameters from all clients. It stacks
    the parameters for each layer across clients, computes the mean, and returns
    the averaged parameters. The @torch.no_grad decorator ensures that this operation
    is not tracked by autograd.

    Args:
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.

    Returns:
        list: A list of averaged parameters which lies on `DEVICE`.
    """
    average_params = [
        torch.mean(torch.stack(param_list), dim=0)
        for param_list in zip(*client_train_params_dict.values())
    ]

    return average_params


@torch.no_grad
def compute_metrics(model, ds_path, ds_name, test_ds_dict):
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
        test_ds_dict (Dict): A dictionary with key the name of the test_ds and value the DataLoader for this test dataset.

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """

    def _compute_metrics(_model, _ds_path, _config_name, _test_ds):

        # Load the evaluation metric
        metric = evaluate.load(path=_ds_path, config_name=_config_name)

        testing_loss = 0.0
        num_batches = len(_test_ds)

        # Set the model to evaluation mode
        _model.eval()

        for batch in _test_ds:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Perform a forward pass
            outputs = _model(**batch)
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

    if ds_path == 'glue':
        if ds_name == 'mnli':
            metrics_m = _compute_metrics(model, ds_path, 'mnli_matched', test_ds_dict['mnli_matched'])
            metrics_m = {k + "_m": v for k, v in metrics_m.items()}  # Rename keys

            metrics_mm = _compute_metrics(model, ds_path, 'mnli_mismatched', test_ds_dict['mnli_mismatched'])
            metrics_mm = {k + "_mm": v for k, v in metrics_mm.items()}  # Rename keys

            return {**metrics_m, **metrics_mm}
        else:
            return _compute_metrics(model, ds_path, ds_name, test_ds_dict[ds_name])


@torch.no_grad
def compute_drifts(old_params, new_params):
    """
    Calculate the drift (difference) between old and new parameters.

    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        new_params (list of torch.nn.Parameter): The updated parameters.

    Returns:
        list of torch.Tensor: The computed drifts for each parameter stored in SAVE_DEVICE.
    """

    drifts = []

    for old_param, new_param in zip(old_params, new_params):
        old_param = old_param.to(DEVICE)
        new_param = new_param.to(DEVICE)

        # TODO: Maybe this `SAVE_DEVICE` can move down to the `compute_client_drifts` function. THe idea should be
        # TODO: that functions that return list of client stuff should place the stuff on `SAVE_DEVICE`. But,
        # TODO: functions that return single stuff should (not care and) place the stuff on `DEVICE`.
        drifts.append(
            (new_param - old_param).to(SAVE_DEVICE)
        )

    return drifts


"""
return [
    new_param - old_param
    for old_param, new_param in zip(old_params, new_params)
]
"""


@torch.no_grad
def compute_client_drifts(old_params, client_train_params_dict):
    """
    Compute the drifts for all clients based on the original parameters.

    Args:
        old_params (list of torch.nn.Parameter): The original parameters.
        client_train_params_dict (dict): Dictionary of client IDs and their corresponding parameters.

    Returns:
        dict: A dictionary where keys are client IDs and values are lists of drifts for each parameter stored in DEVICE_SAVE.
    """
    return {
        client_id: compute_drifts(old_params, client_params)
        for client_id, client_params in client_train_params_dict.items()
    }


@torch.no_grad
def compute_pseudo_gradients(client_drifts):
    """
    Compute the pseudo-gradient based on the drifts between old and client parameters.

    Args:
        client_drifts (dict): Dictionary of client IDs and their corresponding parameter drifts.

    Returns:
        list of torch.Tensor: The computed pseudo-gradient with the result lying on DEVICE.
    """
    # Average the drifts with the result lying on `DEVICE`
    average_drifts = average_client_parameters(client_drifts)

    pseudo_gradient = [-drift for drift in average_drifts]

    return pseudo_gradient


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
        device = param.device
        gradient = gradient.to(device)

        param.grad = gradient


@torch.no_grad
def vectorize(parameters):
    """
    Concatenates a list of parameter tensors into a single vector.

    Args:
        parameters (list of torch.nn.Parameter): An iterable of parameter tensors.

    Returns:
        torch.Tensor: A single vector containing all the elements of the input parameters with the
                      result lying on `DEVICE`.
    """
    con = []
    for param in parameters:
        param = param.to(DEVICE)
        con.append(param.view(-1))

    return torch.cat(con)

    #return torch.cat([param.view(-1) for param in parameters])


@torch.no_grad
def variance(client_drifts):
    """
    Computes the variance of the client models utilizing the drifts (see paper)

    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).

    Returns:
        float: The computed variance of the client drifts.
    """

    """
    # Vectorize each client's drifts
    drifts_vecs = [vectorize(drifts) for drifts in client_drifts.values()]
    # Compute the squared l2 norms of each client's drifts
    norm_sq_drifts = [torch.dot(vec, vec) for vec in drifts_vecs]
    # Compute the average of the squared norms of the individual client drifts
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)
    """
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        # Vectorize the client-drifts with the result lying on `DEVICE`
        drift_vec = vectorize(drifts)
        # Compute the squared l2 norm of the client-drift, with the result lying on `DEVICE`
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        # Append the squared l2 norm to the list
        norm_sq_drifts.append(norm_sq_drift)
    # Compute the average of the squared norms of the individual client drifts
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)

    # Compute the average drift, with the result lying on `DEVICE`
    avg_drift = average_client_parameters(client_drifts)
    # Vectorize the average drift, with the result lying on `DEVICE`
    avg_drift_vec = vectorize(avg_drift)
    # Compute the squared l2 norm of the average drift, with the result lying on `DEVICE`
    norm_sq_avg_drift = torch.dot(avg_drift_vec, avg_drift_vec)

    # variance of the client models
    var = avg_norm_sq_drifts - norm_sq_avg_drift

    return var.item(), avg_norm_sq_drifts.item(), norm_sq_avg_drift.item()

def fda_sketch_estimation(client_drifts, ams_sketch):
    """
    Compute the linear estimation of ||avg(u_t)||^2 using SketchFDA. The way we compute it is by its equivalent form
    which is (1/1+e) * M_2(sk(avg(u_t))). Of course, in a real system implementation we would first compute each drift's
    sketch and then average the sketches on the server. They are equivalent, we simply do this because it is more
    efficient. (see paper)

    Args:
        client_drifts (dict): A dictionary where keys are client IDs and values are lists of parameter tensors (drifts).
        ams_sketch (AmsSketch): The AMS sketch used to compute the sketch estimation.
    Returns:
        float: The linear estimation of ||avg(u_t)||^2.
    """

    # Compute the average drift as a vector avg(u_t), with the result lying on `DEVICE`
    avg_drift = vectorize(average_client_parameters(client_drifts))

    # Compute the sketch of the average drift, with the result lying on `DEVICE`
    sk = ams_sketch.sketch_for_vector(avg_drift)
    # Save the epsilon value for the sketch
    epsilon = ams_sketch.epsilon
    # Compute the approximation of ||avg(u_t)||^2 using the sketch strategy
    est = (1 / (1+epsilon)) * ams_sketch.estimate_euc_norm_squared(sk)

    return est

def fda_variance_approx(client_drifts, ams_sketch=None):

    norm_sq_drifts = []
    for drifts in client_drifts.values():
        # Vectorize the client-drifts with the result lying on `DEVICE`
        drift_vec = vectorize(drifts)
        # Compute the squared l2 norm of the client-drift, with the result lying on `DEVICE`
        norm_sq_drift = torch.dot(drift_vec, drift_vec)
        # Append the squared l2 norm to the list
        norm_sq_drifts.append(norm_sq_drift)
    # Compute the average of the squared norms of the individual client drifts
    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)

    # Compute the approximation of ||avg(u_t)||^2 using the sketch strategy
    norm_sq_avg_drift_approx = fda_sketch_estimation(client_drifts, ams_sketch)

    # variance approximation of the client models
    var_approx = avg_norm_sq_drifts - norm_sq_avg_drift_approx

    return var_approx.item()


@torch.no_grad
def get_updated_client_parameters(client_train_params_dict, sampled_clients, params):
    """
    Update the parameters of the sampled clients with the current global parameters.

    This function ensures there are no ID collisions or overwriting issues by creating a new dictionary
    for the updated client parameters.

    Args:
        client_train_params_dict (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        sampled_clients (list): A list of newly sampled client IDs.
        params (list of torch.nn.Parameter): The current global parameters to be assigned to the sampled clients.

    Returns:
        dict: Dictionary where keys are client IDs and values are lists of parameter tensors.
    """

    new_client_train_params_dict = dict()

    for new_client_id, old_params in zip(sampled_clients, client_train_params_dict.values()):
        # Copy the global parameters to the new client's parameters
        copy_parameters(
            from_params=params,
            to_params=old_params
        )

        # Assign the updated parameters to the new client ID in the new dictionary.
        new_client_train_params_dict[new_client_id] = old_params

    return new_client_train_params_dict


def stop_training_with_threshold(hyperparams, metrics):

    if hyperparams['checkpoint'] == 'roberta-base':

        if hyperparams['ds_name'] == 'mrpc':
            if metrics['accuracy'] >= 0.902 * 0.95:
                return True

        if hyperparams['ds_name'] == 'rte':
            if metrics['accuracy'] >= 0.789 * 0.95:
                return True

        if hyperparams['ds_name'] == 'cola':
            if metrics['matthews_correlation'] >= 0.636 * 0.95:
                return True

        if hyperparams['ds_name'] == 'sst2':
            if metrics['accuracy'] >= 0.948 * 0.99:
                return True

        if hyperparams['ds_name'] == 'qnli':
            if metrics['accuracy'] >= 0.928 * 0.95:
                return True

    return False
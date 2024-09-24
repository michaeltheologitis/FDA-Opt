from fdaopt.utils import DEVICE
from fdaopt.models.ops import copy_parameters

def federated_training_step(model, train_params, client_train_params, client_opt, fed_ds):
    """
    Performs a single federated training step.

    This function trains each client starting with its client-specific model parameters, updates them
    with the client optimizer and client-specific batch, and returns. At the end, client_train_params
    have the updated client-specific parameters (after training on their specific batch).

    Args:
        model (torch.nn.Module): The model to be trained.
        train_params (list): A list of trainable parameters of the model.
        client_train_params (dict): A dictionary where keys are client IDs and values are lists of parameter tensors.
        client_opt (torch.optim.Optimizer): The optimizer for updating the model parameters.
        fed_ds (FederatedDataset): An object that provides the batches for each client.
    Returns:
        float: The average training loss across all clients for this federated training step.
    """

    # Set the model to training mode
    model.train()

    training_loss = 0.0
    num_clients = len(client_train_params)

    # Iterate over each client
    for client_id in client_train_params.keys():
        # Copy client-specific parameters to the model's parameters
        copy_parameters(
            from_params=client_train_params[client_id],
            to_params=train_params
        )

        # Get the next batch of data for the current client
        batch = fed_ds.next_client_batch(client_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Perform a forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        client_opt.step()

        # Copy the updated model parameters back to the client's parameter set
        copy_parameters(
            from_params=train_params,
            to_params=client_train_params[client_id]
        )

        # Zero the gradients before the next backward pass
        client_opt.zero_grad()

        # Accumulate the loss
        training_loss += loss.item()

    # Calculate the average loss
    training_loss = training_loss / num_clients

    return training_loss

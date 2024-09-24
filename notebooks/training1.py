import torch
import numpy as np
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import random

import gc

import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

DEVICE = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

def calculate_prior_distribution(raw_train_dataset):
    """
    Calculate the prior distribution of labels in the training dataset.

    This function calculates the proportion of each label in the training dataset, 
    which is used as the prior distribution for Dirichlet sampling.

    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.

    Returns:
        tuple: A tuple containing:
            - prior_distribution (numpy.ndarray): An array of label proportions.
            - num_labels (int): The number of unique labels in the dataset.
    """
        
    # Calculate the number of samples for each label
    label_count_dict = dict(raw_train_dataset.to_pandas()['label'].value_counts())
    # Num. of samples
    n = len(raw_train_dataset) 
    # (label, count) list, sorted with label increasing
    label_count_tuple = sorted(label_count_dict.items(), key=lambda x: x[0])
    # prior distribution
    prior_distribution = np.array([c / n for _, c in label_count_tuple])
    
    # number of labels
    num_labels = len(label_count_dict)
    
    return prior_distribution, num_labels


def sample_to_client(client_sample_counts, sample_label):
    """
    Assign a sample to a client based on the sample's label and the remaining counts for each client.

    This function assigns a sample to the client that still needs more samples of the given label.
    It decrements the count for that label for the chosen client. If no client has a remaining count
    of at least 1 for the label, the function assigns the sample to the client with the largest leftover count
    for that label.

    Args:
        client_sample_counts (numpy.ndarray): A 2D array where each row corresponds to a client and each column corresponds
            to the count (float) of samples needed for each label. The element at (i, j) represents the decimal number of 
            samples of label j that client i still needs.
        sample_label (int): The label of the sample to be assigned to a client.
        
    Returns:
        int: The index of the client to which the sample has been assigned.
    """
    num_clients = len(client_sample_counts)
    client_indices = np.random.permutation(num_clients)
    
    for client_idx in client_indices:
        client_sample_count = client_sample_counts[client_idx]
        
        if client_sample_count[sample_label] >= 1:
            client_sample_count[sample_label] -= 1
            
            return client_idx
    
    # if all client data counts are less than 1, then assign the sample to the largest leftover
    client_idx = np.argmax(client_sample_counts[:, sample_label])
    client_sample_counts[client_idx][sample_label] -= 1
    
    return client_idx


def federated_dirichlet_datasets(raw_train_dataset, prior_distribution, num_clients, alpha):
    """
    Create federated datasets using Dirichlet-distributed label partitions.

    This function partitions the training dataset into multiple subsets, each corresponding 
    to a client. The label distribution for each client is drawn from a Dirichlet distribution 
    parameterized by the given prior distribution and concentration parameter alpha.
    The function also aims to keep the datasets as equal in size as possible.

    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.
        prior_distribution (numpy.ndarray): An array representing the prior distribution of labels.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter for the Dirichlet distribution.

    Returns:
        list: A list of `datasets.arrow_dataset.Dataset` objects, each representing a client's dataset.
    """
    
    # Num. of samples
    n = len(raw_train_dataset)
    
    client_num_samples = n / num_clients
    
    # Draw label distributions for each client from Dirichlet distribution
    # Each i-th row represents the distribution of labels for the i-th client
    client_distributions = np.random.dirichlet(alpha * prior_distribution, num_clients)
    
    # Initialize client data indices
    client_sample_indices = [[] for i in range(num_clients)]
    
    # Calculate the number of samples per label each client should have
    # client_sample_counts[i] is an array of `num_label` elements
    # -- the counts for each label for the i-th client.
    client_sample_counts = np.array([
        client_distributions[client_idx, :] * client_num_samples
        for client_idx in range(num_clients)
    ])
    
    for sample_idx, sample in enumerate(raw_train_dataset):
        client_idx = sample_to_client(client_sample_counts, sample['label'])

        client_sample_indices[client_idx].append(sample_idx)
        
    # Create a Dataset for each client
    client_datasets = []
    for client_indices in client_sample_indices:
        client_dataset = raw_train_dataset.select(client_indices)
        client_datasets.append(client_dataset)
        
    return client_datasets


def tokenize_client_datasets(client_datasets, tokenize_fn):
    """
    Tokenize and preprocess a list of client datasets.

    This function tokenizes and preprocesses each dataset in the provided list of client datasets.
    It applies the specified tokenization function, renames the "label" column to "labels",
    removes unnecessary columns, and sets the format to PyTorch tensors.

    Args:
        client_datasets (list of datasets.arrow_dataset.Dataset): A list of client datasets to be tokenized and preprocessed.
        tokenization_fn (function): A function that takes an example and returns its tokenized form.

    Returns:
        list of datasets.arrow_dataset.Dataset: A list of tokenized and preprocessed client datasets.
    """
    
    # Define the expected columns
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    
    tok_client_datasets = []

    for client_dataset in client_datasets:
        tok_client_dataset = client_dataset.map(tokenize_fn, batched=True)

        tok_client_dataset = tok_client_dataset.rename_column("label", "labels")

        # Identify columns to remove
        columns_to_remove = [
            column for column in tok_client_dataset.column_names 
            if column not in expected_columns
        ]

        # Remove unnecessary columns
        tok_client_dataset = tok_client_dataset.remove_columns(columns_to_remove)

        # Set the format to PyTorch tensors
        tok_client_dataset.set_format("torch")

        # Add the processed dataset to the list
        tok_client_datasets.append(tok_client_dataset)
        
    return tok_client_datasets


def preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, batch_size):
    """
    Preprocess and tokenize the test dataset, then create a DataLoader for it.

    This function tokenizes and preprocesses the provided test dataset using the specified
    tokenization function. It renames the "label" column to "labels", removes unnecessary columns,
    sets the format to PyTorch tensors, and then creates a DataLoader for the test dataset.

    Args:
        raw_test_dataset (datasets.arrow_dataset.Dataset): The raw test dataset to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.
        data_collator (transformers.DataCollator): A data collator to be used for padding and batching.
        batch_size (int): The batch size to be used by the DataLoader.

    Returns:
        DataLoader: A DataLoader for the tokenized and preprocessed test dataset.
    """
    
    # Define the expected columns
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    
    tok_test_dataset = raw_test_dataset.map(tokenize_fn, batched=True)
    
    tok_test_dataset = tok_test_dataset.rename_column("label", "labels")
    
    # Identify columns to remove
    columns_to_remove = [
        column for column in tok_test_dataset.column_names 
        if column not in expected_columns
    ]
    
    # Remove unnecessary columns
    tok_test_dataset = tok_test_dataset.remove_columns(columns_to_remove)
    
    # Set the format to PyTorch tensors
    tok_test_dataset.set_format("torch")
    
    # Create a DataLoader for the test dataset
    test_ds = DataLoader(
        tok_test_dataset, batch_size=batch_size, collate_fn=data_collator
    )
        
    return test_ds


def create_client_dataloaders(tok_client_datasets, batch_size, collate_fn):
    """
    Create data loaders for a list of tokenized client datasets.

    This function takes a list of tokenized client datasets and creates a DataLoader for each dataset.
    The resulting data loaders are stored in a list and returned.

    Args:
        tok_client_datasets (list of datasets.arrow_dataset.Dataset): A list of tokenized client datasets.
        batch_size (int): The batch size to be used by the data loaders.
        collate_fn (function): A collate function to be used by the data loaders.

    Returns:
        list of DataLoader: A list of data loaders, each corresponding to a tokenized client dataset.
    """
    client_dataloaders = []

    for tok_client_dataset in tok_client_datasets:
        client_dataloader = DataLoader(
            tok_client_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn
        )
        
        client_dataloaders.append(client_dataloader)
        
    return client_dataloaders


def tokenize_function(ds_path, ds_name, tokenizer):
    """
    Return a tokenization function based on the dataset path and name.

    This function returns the appropriate tokenization function for the specified dataset.
    Currently, it supports the GLUE MRPC dataset.

    Args:
        ds_path (str): The path or identifier of the dataset.
        ds_name (str): The name of the dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used.

    Returns:
        function: A tokenization function.
    """
    
    if ds_path == "glue" and ds_name == "mrpc":
        return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    return None


def test_dataset_split(raw_datasets, ds_path):
    """
    Split the raw dataset into the appropriate test set.

    This function extracts the test set from the provided raw dataset based on the dataset path.
    For the GLUE dataset, it returns the validation split as the test set following prior work.

    Args:
        raw_datasets (datasets.DatasetDict): The raw dataset containing multiple splits.
        ds_path (str): The path or identifier of the dataset.

    Returns:
        datasets.Dataset: The extracted test dataset split.
    """
    
    if ds_path == 'glue':
        return raw_datasets['validation']
    

def prepare_federated_datasets(ds_path, ds_name, checkpoint, num_clients, alpha, batch_size):
    """
    Prepare federated datasets and create corresponding DataLoaders.

    This function handles the entire process of loading the raw dataset, partitioning it into
    training federated datasets using Dirichlet distribution, tokenizing the datasets, and creating
    DataLoaders for each client's dataset. It also tokenizes, and creates a dataloader for the test dataset.

    Args:
        ds_path (str): The path or identifier of the dataset.
        ds_name (str): The name of the dataset.
        checkpoint (str): The checkpoint identifier for the tokenizer.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter for the Dirichlet distribution.
        batch_size (int): The batch size to be used by the DataLoaders.

    Returns:
        tuple: A tuple containing:
            - FederatedDataset: An object containing client DataLoaders for federated learning.
            - DataLoader: A DataLoader for the tokenized and preprocessed test dataset.
    """
    
    # Load the raw dataset
    raw_datasets = load_dataset(path=ds_path, name=ds_name)
    
    # 1. Test dataset
    raw_test_dataset = test_dataset_split(raw_datasets, ds_path)
    # 2. Training dataset
    raw_train_dataset = raw_datasets['train']

    # Calculate the prior distribution
    prior_distribution, num_labels = calculate_prior_distribution(raw_train_dataset)

    # Partition the dataset into federated datasets
    client_datasets = federated_dirichlet_datasets(raw_train_dataset, prior_distribution, num_clients, alpha)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Create the tokenization function
    tokenize_fn = tokenize_function(ds_path, ds_name, tokenizer)

    # Tokenize the client datasets
    tok_client_datasets = tokenize_client_datasets(client_datasets, tokenize_fn)

    # Create DataLoaders for each client dataset
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, data_collator)
    
    fed_ds = FederatedDataset(client_dataloaders)

    # Preprocess the test dataset
    test_ds = preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, batch_size)

    return fed_ds, test_ds


class FederatedDataset:
    """
    A class to handle federated datasets for training in a federated learning setup.

    This class encapsulates the logic for managing multiple client dataloaders and providing
    batched data for federated learning training loops.

    Args:
        client_dataloaders (list of DataLoader): A list of DataLoader objects, each corresponding to a client's dataset.
    """
    
    def __init__(self, client_dataloaders):
        self.client_dataloaders = client_dataloaders
        self.client_batch_generators = [
            self.dataloader_batch_generator(client_dl)
            for client_dl in client_dataloaders
        ]
    
    def epoch_steps(self, client_ids):
        """
        Determine the number of steps (batches) in an epoch for the given clients.

        Args:
            client_ids (list of int): A list of client IDs.

        Returns:
            int: The maximum number of steps (batches) for the given clients.
        """
        
        return max(
            len(self.client_dataloaders[client_id])
            for client_id in client_ids
        )
    
    def next_client_batch(self, client_id):
        """
        Retrieve the next batch of data for the specified client.

        Args:
            client_id (int): The ID of the client.

        Returns:
            dict: A batch of data from the client's DataLoader.
        """
        
        return next(
            self.client_batch_generators[client_id]
        )
    
    @staticmethod
    def dataloader_batch_generator(dataloader):
        """
        A generator that yields (unending) batches of data from a DataLoader.

        Args:
            dataloader (DataLoader): A DataLoader object.

        Yields:
            dict: A batch of data from the DataLoader.
        """
        
        while True:
            for batch in dataloader:
                yield batch
                
                

from transformers import AutoModelForSequenceClassification
from torch.optim import SGD, Adam

import evaluate



@torch.no_grad
def copy_parameters(from_params, to_params):
    """
    Copies the values from one set of parameters to another.

    This function operates in-place and modifies the `to_parameters` directly.
    The @torch.no_grad() decorator ensures that this operation is not tracked 
    by autograd, preventing unnecessary computation and memory usage.

    Args:
        from_parameters (list of torch.nn.Parameter): An iterable of source parameters to copy from.
        to_parameters (list of torch.nn.Parameter): An iterable of destination parameters to copy to.
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
    old_clients = list(client_train_params.keys())
    
    # Iterate over pairs of new client IDs and old client IDs
    for new_client_id, old_client_id in zip(sampled_clients, old_clients):
        # Reassign the entry in client_train_params dictionary
        client_train_params[new_client_id] = client_train_params.pop(old_client_id)
        
        # Copy the global parameters to the new client's parameters
        copy_parameters(
            from_params=params,
            to_params=client_params[new_client_id]
        )
        

class ClientSampler:
    """
    A class that helps random sampling of clients in each round.

    Attributes:
        client_ids (list): A list of client IDs to be sampled.
        sample_size (int): The size of each group of sampled client IDs.
        num_clients (int): The total number of clients.
        client_sample_generator (generator): A generator that yields groups of client IDs.
    """
    
    def __init__(self, client_ids, sample_size):
        self.client_ids = client_ids
        self.sample_size = sample_size
        self.num_clients = len(client_ids)
        
        if self.num_clients % self.sample_size != 0:
            raise ValueError("The number of clients must be divisible with the sample size.")
        
        self.client_sample_generator = self.sample_generator()

    def sample(self):
        """
        Returns the next group of sampled client IDs.

        Returns:
            list: A list of sampled client IDs of length `sample_size`.
        """
        return next(self.client_sample_generator)
        
    def sample_generator(self):
        """
        A generator function to continuously generate groups of sampled client IDs.

        This function shuffles the client IDs and partitions them into groups of the specified sample size.
        It yields each group of sampled client IDs.

        Yields:
            list: A list of sampled client IDs of length `sample_size`.
        """
        
        while True:
            # Shuffle client_ids without modifying the original list https://stackoverflow.com/a/47750824
            shuffled_client_ids = random.sample(self.client_ids, self.num_clients)

            # Partition the shuffled list into groups of the specified sample size
            groups = [shuffled_client_ids[i:i + self.sample_size] for i in range(0, self.num_clients, self.sample_size)]

            # Yield each group of sampled client IDs
            for sampled_client_ids in groups:
                yield sampled_client_ids

                
def get_opt_hyperparams(where, hyper_params):
    """
    Extracts optimizer hyperparameters from the given hyperparameters dictionary.

    Args:
        where (str): A prefix indicating whether the optimizer is 'server' or 'client'.
        hyperparams (dict): A dictionary containing hyperparameters including optimizer hyperparameters.

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
    
    
class MetricsHandler:
    """
    A class to handle the storage and management of metrics for federated learning experiments.

    This class allows the user to store hyperparameters, append metrics for each round, and save the
    metrics to a CSV file.

    Attributes:
        hyperparams (dict): A dictionary to store hyperparameters.
        filename (str): The filename generated from the hyperparameters.
        round_metrics (list): A list to store metrics for each round.
    """
    
    def __init__(self, **hyperparams):
        """
        Initializes the MetricsHandler class with provided hyperparameters.

        Args:
            **hyperparams: Arbitrary keyword arguments for hyperparameters.
        """
        self.hyperparams = hyperparams
        
        # Generate a unique filename based on hyperparameters
        self.filename = "_".join([f"{str(v)}" for k, v in self.hyperparams.items()]) + '.csv'
        
        # Initialize an empty list to store round metrics
        self.round_metrics = []
        
                    
    def append_round_metrics(self, metrics_dict):
        """
        Appends the metrics of the current round to the round_metrics list.

        Args:
            metrics_dict (dict): A dictionary containing the metrics for the current round.
        """
        self.round_metrics.append(metrics_dict)
    
    def save_metrics(self, filepath='./'):
        """
        Saves all the metrics to a CSV file.

        This method combines hyperparameters and round metrics into a DataFrame and saves it as a CSV file.

        Args:
            filepath (str, optional): The directory where the CSV file will be saved. Defaults to './'.
        """
        # Combine the unique hyperparameters with round metrics
        round_metrics_with_id = [
            dict(self.hyperparams.items() | round_metrics.items())
            for round_metrics in self.round_metrics
        ]
        
        # Create a DataFrame with columns the hyperparameters (i.e. the test-id) and round metrics
        df = pd.DataFrame(
            round_metrics_with_id, 
            columns=list(self.hyperparams.keys())+list(self.round_metrics[0].keys())
        )
        
        # Save the DataFrame to a CSV file
        df.to_csv(filepath + self.filename)
        
        

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
    'server_opt_name': 'SGDM',
    'server_opt_lr': 0.33,
    'server_opt_momentum': 0.9,
    'client_opt_name': 'SGD',
    'client_opt_lr': 0.001
}

# Prepare federated datasets and DataLoaders
fed_ds, test_ds = prepare_federated_datasets(hyperparams['ds_path'], hyperparams['ds_name'], hyperparams['checkpoint'], hyperparams['num_clients'], hyperparams['alpha'], hyperparams['batch_size'])

model = AutoModelForSequenceClassification.from_pretrained(hyperparams['checkpoint'], num_labels=hyperparams['num_labels'])

model = model.to(DEVICE)

# Extract trainable parameters from the model, which reside on the device that the model resides in
train_params = [param for param in model.parameters() if param.requires_grad]

# Create a copy of the trainable parameters, detached from the computation graph and moved to the CPU
# TODO: This hardcodes 'cpu' as the target device for the detached parameters
round_start_train_params = [param.detach().clone() for param in train_params]

#server_opt = Adam(train_params, **get_opt_hyperparams('server', hyperparams))
server_opt = SGD(train_params, **get_opt_hyperparams('server', hyperparams))
client_opt = SGD(train_params, **get_opt_hyperparams('client', hyperparams))  # Note: One optimizer because SGD is stateless.
print(server_opt)
print(client_opt)

client_ids = list(range(hyperparams['num_clients']))

client_sampler = ClientSampler(client_ids, hyperparams['clients_per_round'])

metrics_lst = []

client_train_params = {
    client_id: [param.detach().clone() for param in round_start_train_params]
    for client_id in range(hyperparams['clients_per_round'])
}

metrics_handler = MetricsHandler(**hyperparams)

for r in range(hyperparams['total_rounds']):
    
    training_loss = 0.0

    # Save the model parameters at the start of this round
    sampled_clients = client_sampler.sample() 
    
    # Save the model parameters at the start of this round
    copy_parameters(
        from_params=train_params,
        to_params=round_start_train_params
    )

    # Initialize a dictionary to store the trainable parameters for each client
    # Each client's parameters are cloned from the round start parameters
    #client_train_params = {
    #    client_id: [param.detach().clone() for param in round_start_train_params]
    #    for client_id in sampled_clients
    #}
        
    update_sampled_client_parameters(client_train_params, sampled_clients, round_start_train_params)

    # Calculate the total number of steps for this epoch
    epoch_steps = hyperparams['local_epochs'] * fed_ds.epoch_steps(sampled_clients)

    for step in range(epoch_steps):
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
    
    # Calculate evaluation metrics on the test set
    metrics = {"round": r+1} | compute_metrics(model, hyperparams['ds_path'], hyperparams['ds_name'], test_ds)
    # Calculate the average training loss for the round
    metrics['training_loss'] = training_loss / epoch_steps
    # Calculate variance and helpful metrics
    metrics['variance'], metrics['avg_norm_sq_drifts'], metrics['norm_sq_avg_drift'] = variance(client_drifts)
    # Pass round metrics to handler
    metrics_handler.append_round_metrics(metrics)
    
    print(metrics)
    
    gc.collect()
        
metrics_handler.save_metrics('./results/')
import pandas as pd
import os

# Get the directory of the current script (main.py)
script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../../results'
ROUND_METRICS_PATH = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/round_metrics'))

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

    def __init__(self, hyperparams):
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

    def save_metrics(self, filepath=ROUND_METRICS_PATH):
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
            columns=list(self.hyperparams.keys()) + list(self.round_metrics[0].keys())
        )

        # Save the DataFrame to a CSV file
        df.to_csv(f"{filepath}/{self.filename}")

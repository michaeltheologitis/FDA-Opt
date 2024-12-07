import re
import subprocess
import argparse

def modify_learning_rates(command, new_server_lr, new_client_lr):
    """
    Modify the value of 'lr' in 'server_opt_args' and 'client_opt_args' in a terminal command string.

    Args:
        command (str): The original command string.
        new_server_lr (float): The new learning rate for the server.
        new_client_lr (float): The new learning rate for the client.

    Returns:
        str: The updated command string.
    """
    # Regular expressions for server and client learning rates
    server_lr_pattern = r"(--server_opt_args\s.*?)(lr=\d*\.?\d*)"
    client_lr_pattern = r"(--client_opt_args\s.*?)(lr=\d*\.?\d*)"
    
    # Replace the server 'lr' value with the new value
    command = re.sub(server_lr_pattern, lambda match: f"{match.group(1)}lr={new_server_lr}", command)
    # Replace the client 'lr' value with the new value
    command = re.sub(client_lr_pattern, lambda match: f"{match.group(1)}lr={new_client_lr}", command)
    
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parse space-separated lists of floats
    parser.add_argument('--server_lrs', type=float, nargs='+', required=True, help="Space-separated list of server learning rates")
    parser.add_argument('--client_lrs', type=float, nargs='+', required=True, help="Space-separated list of client learning rates")

    args = parser.parse_args()

    # Input template command
    command = input("Enter template command (copy-paste): ")

    # Iterate over learning rates and modify the command
    for server_lr in args.server_lrs:
        for client_lr in args.client_lrs:
            tmp_command = modify_learning_rates(command, server_lr, client_lr)
            print(f"{tmp_command}")
            subprocess.run(tmp_command, shell=True, text=True, capture_output=True)
import subprocess
import signal
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, './hyperparameters'))
OUTPUT_DIR = os.path.normpath(os.path.join(script_directory, './results/output'))

processes = []

def signal_handler(signum, frame):
    """
    Signal handler to kill all child processes.
    """
    global processes  # Explicitly state that we're using the global variable

    for proc in processes:
        if not proc.poll():  # Check if the process is still running
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()


if __name__ == '__main__':

    files = os.listdir(HYPERPARAM_DIR)

    while os.listdir(HYPERPARAM_DIR):

        hyperparam_file = os.listdir(HYPERPARAM_DIR)[0]

        # Ask the user whether to start a new process or quit
        user_response = input(
            f"Will start {hyperparam_file}. Type the DEVICE to run (e.g. 'cuda:0', 'cpu'): "
        ).strip().lower()

        # Construct the command
        cmd = [
            'python', '-u', '-m', 'fdaopt.main', f'--filename={hyperparam_file}', f'--device={user_response}'
        ]

        hyperparam_id = hyperparam_file.split('.')[0]
        with open(f'{OUTPUT_DIR}/{hyperparam_id}.out', 'w') as stdout_file:
            with open(f'{OUTPUT_DIR}/{hyperparam_id}.err', 'w') as stderr_file:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
                processes.append(process)

        print()

        #os.remove(f"{HYPERPARAM_DIR}/{hyperparam_file}")

    print(f"All available hyper-parameter files have been started.")
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all child processes to complete
    for p in processes:
        p.wait()

    print("All processes terminated. Exiting.")


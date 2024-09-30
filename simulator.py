import subprocess
import signal
import os
import argparse
from collections import namedtuple
import time

script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, './hyperparameters'))
OUTPUT_DIR = os.path.normpath(os.path.join(script_directory, './results/output'))

# Declaring namedtuple()
Task = namedtuple('Task', ['hyperparam_file', 'hyperparam_id', 'stdout_filename', 'stderr_filename'])

processes_devices, task_queue = [], []

device_counts, device_limits = {}, {}

def signal_handler(signum, frame):
    """
    Signal handler to kill all child processes.
    """
    global processes_devices  # Explicitly state that we're using the global variable

    for (proc, device) in processes_devices:
        if not proc.poll():  # Check if the process is still running
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()

def set_device_limits(device_limit_args):
    global device_limits

    device_limits = {}

    for arg in device_limit_args:
        device, num = arg.split('=')
        device_limits[device] = int(num)


def get_free_device():
    global device_counts, device_limits

    for device, count in device_counts.items():
        if count < device_limits[device]:
            return device

    return None


def check_finished_processes():
    global processes_devices, device_counts

    for (proc, device) in processes_devices[:]:  # Iterate over a copy of the list
        if proc.poll() is not None:  # Process has finished
            device_counts[device] -= 1  # Free a device slot
            processes_devices.remove((proc, device))  # Remove from the list of processes


def terminate_and_quit():
    global processes_devices

    print("Terminating all processes based on user input.")
    for (proc, device) in processes_devices:
        if proc.poll() is None:  # Check if the process is still running
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device_limits', nargs='*', required=True,
                        help="How many tests can be run in-parallel in each device (e.g., --device_limit cuda:0=1 cuda:1=1 cpu=2")

    args = parser.parse_args()

    # Initialize device limits and counts
    set_device_limits(args.device_limits)
    device_counts = {device: 0 for device in device_limits}

    hyperparameter_files = os.listdir(HYPERPARAM_DIR)

    for hyperparam_file in hyperparameter_files:

        # Ask the user whether to start a new process or quit
        user_response = input(
            f"Press 'Enter' to queue the test for {hyperparam_file} or 'q' to terminate all and quit: "
        ).strip().lower()

        # If the user enters 'q', terminate all running processes and exit
        if user_response == "q":
            terminate_and_quit()

        hyperparam_id = hyperparam_file.split('.')[0]

        task = Task(
            hyperparam_file=hyperparam_file,
            hyperparam_id=hyperparam_id,
            stdout_filename=f'{OUTPUT_DIR}/{hyperparam_id}.out',
            stderr_filename=f'{OUTPUT_DIR}/{hyperparam_id}.err'
        )

        task_queue.append(task)

        #os.remove(f"{HYPERPARAM_DIR}/{hyperparam_file}")

    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while task_queue:

        check_finished_processes()
        device = get_free_device()

        if device:
            task = task_queue.pop(0)

            cmd = ['python', '-u', '-m', 'fdaopt.main', f'--filename={task.hyperparam_file}', f'--device={device}']

            # Copy the current environment and add/modify PYTORCH_CUDA_ALLOC_CONF
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


            with open(task.stdout_filename, 'w') as stdout_file:
                with open(task.stderr_filename, 'w') as stderr_file:
                    print(f"Running on {device} the hyperparameter file {task.hyperparam_file}.")
                    process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, env=env)
                    processes_devices.append((process, device))

            device_counts[device] += 1

        else:
            time.sleep(10)

    # Wait for all remaining processes to finish after the task queue is empty
    while processes_devices:
        check_finished_processes()  # Remove finished processes
        time.sleep(10)  # Sleep briefly to avoid busy-waiting

    print("All processes terminated. Exiting.")
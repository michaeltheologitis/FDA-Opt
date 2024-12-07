import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import evaluate

DEVICE = None

# TODO: Decouple
# The DEVICE-RAM program. It can be one of 'low', 'moderate', or 'performance'.
DEVICE_RAM_PROGRAM = 'performance'


def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def set_device(device):
    global DEVICE
    DEVICE = device

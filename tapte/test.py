import numpy as np
import math
import os
from tqdm import tqdm
from prettytable import PrettyTable
import h5py
import json
import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import permutations

from tapte.modules import *


def main(options_file_path, model_to_test):
    
    # Load options_file:
    file_path = options_file_path
    try:
        with open(file_path, 'r') as file:
            options_file = json.load(file)
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    # Setup device-agnostic code:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
      num_cuda_devices = torch.cuda.device_count()
      if options_file["gpus"] > num_cuda_devices:
        raise ValueError(f'There are only {num_cuda_devices} GPUs available, but requested {options_file["gpus"]}.')
      else:
        gpu_indices = list(range(options_file["gpus"]))
    else:
      gpu_indices = None
    
    # Setup testing:

    # Initialize dataset
    test_dataset = TAPTEDataset(options_file, split='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=False)

    # Initialize LightningModule
    tapte = TAPTE(options_file)
    #state_dict = torch.load(f"output/{version}/checkpoints/epoch_{epoch}.pth")
    #tapte.load_state_dict(state_dict)

    tapte_lightning = TAPTELightning(options_file, tapte)

    # Initialize Lightning Tester
    tester = L.Trainer(
    max_epochs=1,
    devices=options_file["gpus"],
    accelerator="auto"
)

    # Start testing
    tester.test(tapte_lightning, test_loader)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('options_file_path', type=str, help='Path to the options file')
    parser.add_argument('--model_to_test', type=str, help='Choose which model to test')

    args = parser.parse_args()

    main(args.options_file_path, args.model_to_test)

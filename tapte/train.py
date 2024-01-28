import numpy as np
import math
import os
import h5py
import json
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from itertools import permutations

from tapte.modules import *


def main(options_file_path):
    
    #Load options_file:
    file_path = options_file_path
    try:
        with open(file_path, 'r') as file:
            options_file = json.load(file)
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    #Setup device-agnostic code:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
      num_cuda_devices = torch.cuda.device_count()
      if options_file["gpus"] > num_cuda_devices:
        raise ValueError(f'There are only {num_cuda_devices} GPUs available, but requested {options_file["gpus"]}.')
      else:
        gpu_indices = list(range(options_file["gpus"]))
    else:
      gpu_indices = None
    
    
    #Setup training:  
    data = Data(options_file, 1000)

    train_inputs, temp_inputs, train_assignments, temp_assignments, train_categories, temp_categories = train_test_split(
        data.inputs, data.assignments, data.categories, test_size=1-options_file["training_size"], random_state=42)

    val_inputs, test_inputs, val_assignments, test_assignments, val_categories, test_categories = train_test_split(
        temp_inputs, temp_assignments, temp_categories, test_size=0.5, random_state=42)

    train_dataset = TAPTEDataset(train_inputs, train_assignments, train_categories)
    val_dataset = TAPTEDataset(val_inputs, val_assignments, val_categories)
    test_dataset = TAPTEDataset(test_inputs, test_assignments, test_categories)

    batch_size = options_file["batch_size"]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=options_file["num_of_workers"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=options_file["num_of_workers"], shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=options_file["num_of_workers"], shuffle=False)

    tapte_lightning = TAPTELightning(options_file)
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=2, name="lightning_logs")
    trainer = L.Trainer(devices=options_file["gpus"], accelerator="auto", max_epochs=options_file["epochs"], logger=logger)
    
    #Training:
    trainer.fit(tapte_lightning, train_loader, val_loader)


if __name__ == "__main__":
    # Check if the script is run with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python train.py -options_file_path")
        sys.exit(1)  # Exit with an error code

    # Extract command line arguments
    arg1 = sys.argv[1]

    # Call the main function with the arguments
    main(arg1)

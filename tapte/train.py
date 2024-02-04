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


def main(options_file_path, checkpoint=None):
    
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
    
    # Setup training:

    # Initialize dataset
    train_dataset = TAPTEDataset(options_file, split='train')
    val_dataset = TAPTEDataset(options_file, split='val')

    train_loader = DataLoader(dataset=train_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=False)

    # Initialize LightningModule
    tapte = TAPTE(options_file)
    tapte_lightning = TAPTELightning(options_file, tapte)

    # Setup output folders
    outputs_folder = 'outputs'
    existing_versions = []
    if outputs_folder in os.listdir("./"):
      existing_versions = [d for d in os.listdir(outputs_folder) if d.startswith('version_')]
      latest_version = max([0] + [int(version.split('_')[1]) for version in existing_versions], default=-1)
      version = latest_version + 1
    else:
      version = 1
    
    version_folder = f'version_{version}'

    # Update the outputs_directory
    outputs_directory = os.path.join(outputs_folder, version_folder)
    if not os.path.exists(outputs_directory):
        os.makedirs(outputs_directory)
    
    # Save options_file
    with open(os.path.join(outputs_directory, "options_file.json"), 'w') as json_file:
        json.dump(options_file, json_file, indent=4)

    # Create TensorBoardLogger
    logger = TensorBoardLogger(save_dir="./", name=outputs_folder, version=version)

    # Create ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(outputs_directory, 'checkpoints'),
        filename="model_{epoch}",
        save_top_k=1,  # Save only the best model
        monitor="val_loss",  # Choose the metric to monitor
        mode="min",  # "min" for validation loss, "max" for validation accuracy
        save_last=True,  # Save the last checkpoint
        verbose=True,
    )

    # Initialize Lightning Trainer
    trainer = L.Trainer(
      max_epochs=options_file["epochs"],
      devices=options_file["gpus"],
      accelerator="auto",
      logger=logger,
      callbacks=[checkpoint_callback],
)   

    # Start training
    print("Options:", end='\n')
    print_dict_as_table(options_file)
    
    if checkpoint:
        checkpoint_path = f"{checkpoint}/checkpoints/last.ckpt"
        trainer.resume_from_checkpoint = checkpoint_path
        print(f"Resuming from {checkpoint_path}")

    trainer.fit(tapte_lightning, train_loader, val_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('options_file_path', type=str, help='Path to the options file')
    parser.add_argument('--checkpoint', type=str, help='Continue from the checkpoint')

    args = parser.parse_args()

    main(args.options_file_path, args.checkpoint)

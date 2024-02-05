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


def main(model_to_test, gpus=None):
    
    # Load options_file:
    file_path = f"{model_to_test}/options_file.json"
    try:
        with open(file_path, 'r') as file:
            options_file = json.load(file)
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    # Setup device-agnostic code:
    if gpus is None:
       if torch.cuda.is_available():
          gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
      num_cuda_devices = torch.cuda.device_count()
      if gpus > num_cuda_devices:
        raise ValueError(f'There are only {num_cuda_devices} GPUs available, but requested {gpus}.')
      else:
        gpu_indices = list(range(gpus))
    else:
      gpu_indices = None
    
    # Setup testing:

    # Initialize dataset
    test_dataset = TAPTEDataset(options_file, split='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=False)

    # Initialize Lightning Tester
    tester = L.Trainer(
    max_epochs=1,
    devices=gpus,
    accelerator="auto",
    default_root_dir=None,
    callbacks=False,
    logger=False 
)
    
    best_epoch = [d for d in os.listdir(f"{model_to_test}/checkpoints/") if d.startswith('best')]
    checkpoint_path = f"{model_to_test}/checkpoints/{best_epoch[0]}"
    tapte = TAPTE(options_file)
    tapte_lightning = TAPTELightning.load_from_checkpoint(checkpoint_path, options_file=options_file, model=tapte)
    tester.test(tapte_lightning, test_loader, ckpt_path=checkpoint_path)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('model_to_test', type=str, help='Choose which model to test')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use for testing')

    args = parser.parse_args()

    main(args.model_to_test, args.gpus)

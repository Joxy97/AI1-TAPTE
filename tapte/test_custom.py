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
    data = Data(options_file, batching=True)
    data.to_device(device)

    train_inputs, temp_inputs, train_assignments, temp_assignments, train_categories, temp_categories = train_test_split(
        data.inputs, data.assignments, data.categories, test_size=1-options_file["split"][0], random_state=42)

    val_inputs, test_inputs, val_assignments, test_assignments, val_categories, test_categories = train_test_split(
        temp_inputs, temp_assignments, temp_categories, test_size=options_file["split"][2]/(options_file["split"][1]+options_file["split"][2]), random_state=42)

    train_inputs, train_assignments, train_categories = shuffle(
        train_inputs, train_assignments, train_categories, random_state=42)

    tapte = TAPTE(options_file)
    #state_dict = torch.load(f"output/{version}/checkpoints/epoch_{epoch}.pth")
    #tapte.load_state_dict(state_dict)

    hsm_loss = HybridSymmetricLoss(options_file).to(device)
    hsm_loss.inference = True

    if torch.cuda.device_count() > 1:
      tapte = nn.DataParallel(tapte, device_ids=gpu_indices)
    tapte.to(device)

    tapte.eval()
    avg_test_loss = 0.0

    test_iterator = tqdm(range(test_inputs.size(0)), desc=f'Testing model', unit='batch')

    for batch in test_iterator:
      with torch.inference_mode():
        predicted_assignment, predicted_category = tapte(test_inputs[batch])
        test_loss = hsm_loss(predicted_assignment, predicted_category, test_assignments[batch], test_categories[batch])
        avg_test_loss += test_loss.item()

    avg_test_loss /= (batch + 1)
    test_iterator.set_postfix(test_loss=avg_test_loss)
    print(f'Average test loss: {avg_test_loss}')
    hsm_loss.print_results()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('options_file_path', type=str, help='Path to the options file')
    parser.add_argument('--model_to_test', type=str, help='Choose which model to test')

    args = parser.parse_args()

    main(args.options_file_path, args.model_to_test)

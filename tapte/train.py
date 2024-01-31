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


def main(options_file_path, manual_training=False):
    
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
    
    #Load data 
    checkpoint = None

    data = Data(options_file, batching=manual_training)

    if manual_training == False:

      outputs_directory = 'outputs'

      # Initialize dataset and data loaders
      data = Data(options_file, batching=manual_training)
      dataset = TAPTEDataset(data.inputs, data.assignments, data.categories, split=options_file["split"])
      train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=True)
      val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=False)
      test_loader = DataLoader(dataset=dataset.test_dataset, batch_size=options_file["batch_size"], num_workers=options_file["num_of_workers"], shuffle=False)

      # Initialize LightningModule
      tapte_lightning = TAPTELightning(options_file)

      # Create TensorBoardLogger
      logger = TensorBoardLogger(save_dir="./", version=None, name=outputs_directory)

      # Create ModelCheckpoint callback
      checkpoint_callback = ModelCheckpoint(
          dirpath=os.path.join(f'{outputs_directory}/{checkpoint}/checkpoints'),
          filename="model_epoch_{epoch}",
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
        callbacks=[checkpoint_callback],  # Use the callback from pytorch_lightning
    )

      # Start training
      print("Options:", end='\n')
      print_dict_as_table(options_file)
      trainer.fit(tapte_lightning, train_loader, val_loader)


    else:
      #Setup:        
      data.to_device(device)

      train_inputs, temp_inputs, train_assignments, temp_assignments, train_categories, temp_categories = train_test_split(
          data.inputs, data.assignments, data.categories, test_size=1-options_file["split"][0], random_state=42)

      val_inputs, test_inputs, val_assignments, test_assignments, val_categories, test_categories = train_test_split(
          temp_inputs, temp_assignments, temp_categories, test_size=options_file["split"][2]/(options_file["split"][1]+options_file["split"][2]), random_state=42)

      train_inputs, train_assignments, train_categories = shuffle(
          train_inputs, train_assignments, train_categories, random_state=42)

      outputs_directory = 'outputs'
      os.makedirs(outputs_directory, exist_ok=True)

      if checkpoint:
          output_directory = os.path.join(outputs_directory, checkpoint)
      else:
          # Find the existing version folders and determine the next version number
          existing_versions = [d for d in os.listdir(outputs_directory) if os.path.isdir(os.path.join(outputs_directory, d))]
          next_version = 1 if not existing_versions else max([int(version.split('_')[1]) for version in existing_versions]) + 1

          # Create the output directory for this run
          output_directory = os.path.join(outputs_directory, f'version_{next_version}')
          os.makedirs(output_directory, exist_ok=True)
          print(f'version_{next_version}')

      # Initialize model, loss function, and optimizer
      tapte = TAPTE(options_file)
      hsm_loss = HybridSymmetricLoss(options_file).to(device)
      optimizer = optim.AdamW(tapte.parameters(), lr=options_file["learning_rate"])

      # Check for existing checkpoints in the version folder
      existing_checkpoints = [f for f in os.listdir(output_directory) if f.startswith('model_epoch_')]
      start_epoch = 0

      # If checkpoints exist, load the latest checkpoint
      if existing_checkpoints:
          latest_checkpoint = max(existing_checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
          checkpoint_path = os.path.join(output_directory, latest_checkpoint)
          tapte.load_state_dict(torch.load(checkpoint_path))
          start_epoch = int(latest_checkpoint.split('_')[2].split('.')[0]) + 1
          print(checkpoint)
          print(f"Resuming training from epoch {start_epoch}")

      if torch.cuda.device_count() > 1:
          tapte = nn.DataParallel(tapte, device_ids=gpu_indices)
      tapte.to(device)

      print(f'Number of GPUs detected on this device: {torch.cuda.device_count()}', end='\n')
      print(f'Currently using GPUs: {gpu_indices}', end='\n')
      print("Options:", end='\n')
      print_dict_as_table(options_file)

      #Training:
      for epoch in range(start_epoch, options_file["epochs"]):
          tapte.train()
          train_loss = 0.0

          train_iterator = tqdm(range(train_inputs.size(0)), desc=f'Epoch {epoch} training', unit='batch')

          for batch in train_iterator:
              assignment, category = tapte(train_inputs[batch])
              loss = hsm_loss(assignment, category, train_assignments[batch], train_categories[batch])
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              train_loss += loss.item()
              avg_train_loss = train_loss / (batch + 1)
              train_iterator.set_postfix(train_loss=avg_train_loss)

          tapte.eval()
          avg_val_loss = 0.0

          val_iterator = tqdm(range(val_inputs.size(0)), desc=f'Epoch {epoch} validation', unit='batch')

          for batch in val_iterator:
              with torch.inference_mode():
                  eval_assignment, eval_category = tapte(val_inputs[batch])
                  eval_loss = hsm_loss(eval_assignment, eval_category, val_assignments[batch], val_categories[batch])

              avg_val_loss += eval_loss.item()

          avg_val_loss /= (batch + 1)
          val_iterator.set_postfix(val_loss=avg_val_loss)

          print(f'Epoch: {epoch}, Average Validation Loss: {avg_val_loss}')

          save_path = os.path.join(output_directory, f'model_epoch_{epoch}.pth')
          torch.save(tapte.state_dict(), save_path)

          if epoch > 0:
            previous_epoch_path = os.path.join(output_directory, f'model_epoch_{epoch - 1}.pth')
            if os.path.exists(previous_epoch_path):
              os.remove(previous_epoch_path)

      print(f'Max number of {options_file["epochs"]} epochs reached!')
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('options_file_path', type=str, help='Path to the options file')
    parser.add_argument('--manual_training', action='store_true', help='Enable manual training')

    args = parser.parse_args()

    main(args.options_file_path, args.manual_training)

import numpy as np
import math
import os
import h5py

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from itertools import permutations


#READ DATASET.H5 AND LOAD INTO TORCH TENSORS:

def load_h5(options_file, size=None, batching=False):
  
  with h5py.File(options_file[f"dataset"], 'r+') as file:
    if size != None:
      num_of_events = size
    else:
      num_of_events = len(h1_data["mask"])

    #Load inputs:
    variables = options_file["variables"]

    inputs_list = []
    assignments_list = []

    for variable in variables:
        inputs_list.append(file[f'INPUTS/Jets/{variable}'][:size])
    inputs_array = np.array(inputs_list)
    inputs = torch.permute(torch.tensor(inputs_array), (1, 2, 0)).to(torch.float)

    #Load assignments:
    h1_data = file['TARGETS']['h1']
    h2_data = file['TARGETS']['h2']
    h3_data = file['TARGETS']['h3']

    assignments = np.zeros((num_of_events, 3, 10, 10), dtype=np.float32)

    mask_h1 = h1_data['mask'][:size].astype(np.float32)
    h1_exists = np.where(mask_h1 == 1)
    mask_h2 = h2_data['mask'][:size].astype(np.float32)
    h2_exists = np.where(mask_h2 == 1)
    mask_h3 = h3_data['mask'][:size].astype(np.float32)
    h3_exists = np.where(mask_h3 == 1)

    indices_h1 = np.stack([h1_data['b1'][h1_exists], h1_data['b2'][h1_exists]])
    indices_h2 = np.stack([h2_data['b1'][h2_exists], h2_data['b2'][h2_exists]])
    indices_h3 = np.stack([h3_data['b1'][h3_exists], h3_data['b2'][h3_exists]])

    perm_indices_h1 = list(permutations(indices_h1))
    perm_indices_h2 = list(permutations(indices_h2))
    perm_indices_h3 = list(permutations(indices_h3))

    assignments[h1_exists, 0, perm_indices_h1[0], perm_indices_h1[1]] = 1
    assignments[h2_exists, 1, perm_indices_h2[0], perm_indices_h2[1]] = 1
    assignments[h3_exists, 2, perm_indices_h3[0], perm_indices_h3[1]] = 1

    assignments = torch.tensor(assignments).to(torch.float)

    #Load categories:
    categories = np.array([mask_h1, mask_h2, mask_h3])
    categories = torch.permute(torch.tensor(categories), (1, 0)).to(torch.float)

    if batching == True:
      batch_size = options_file["batch_size"]
      num_of_batches = int(num_of_events/batch_size)
      inputs = torch.reshape(inputs[:(num_of_batches * batch_size)], (num_of_batches, batch_size, inputs.size(1), inputs.size(2)))
      assignments = torch.reshape(assignments[:(num_of_batches * batch_size)], (num_of_batches, batch_size, assignments.size(1), assignments.size(2), assignments.size(3)))
      categories = torch.reshape(categories[:(num_of_batches * batch_size)], (num_of_batches, batch_size, categories.size(1)))

    return inputs, assignments, categories

class Data(nn.Module):                                              
  def __init__(self, options_file, batching=False):
    super().__init__()
    self.size = options_file["optional_loading_size"]
    if batching == True:
      self.inputs, self.assignments, self.categories = load_h5(options_file, self.size, batching=batching)
    else:
      self.inputs, self.assignments, self.categories = load_h5(options_file, self.size)

  def require_grad(self):
    self.inputs.requires_grad_(True)
    self.assignments.requires_grad_(True)
    self.categories.requires_grad_(True)

  def to_device(self, device):
    self.inputs, self.assignments, self.categories = self.inputs.to(device), self.assignments.to(device), self.categories.to(device)
    
    
#CREATE TORCH DATASET BASED ON LOADED DATA

class TAPTEDataset(Dataset):
    def __init__(self, _inputs, _assignments, _categories, split=[0.8, 0.1, 0.1]):
        # Ensure the split ratios sum to 1
        assert sum(split) == 1.0, "Split ratios should sum to 1.0"

        self.inputs = _inputs
        self.assignments = _assignments
        self.categories = _categories

        # Calculate split sizes
        total_samples = len(self.inputs)
        train_size = int(split[0] * total_samples)
        val_size = int(split[1] * total_samples)
        test_size = total_samples - train_size - val_size

        # Use random_split to split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        assignment = self.assignments[index]
        category = self.categories[index]

        return input, assignment, category
    
    
#BUILDING BLOCKS

class InitialEmbedding(nn.Module):
    def __init__(self, options_file):
        super(InitialEmbedding, self).__init__()
        self.num_of_layers = options_file["num_of_embedding_layers"]
        self.model_dim = options_file["model_dim"]

        # First linear layer
        self.first_layer = nn.Linear(len(options_file["variables"]), self.model_dim)

        # Embedding layers
        layers = []
        for i in range(self.num_of_layers - 1):
            layers.append(nn.Linear(self.model_dim, self.model_dim))
            layers.append(nn.GELU())

        self.embedding_layers = nn.Sequential(*layers)

    def forward(self, input):
        x = nn.functional.gelu(self.first_layer(input))
        x = self.embedding_layers(x)

        return x
    
    
class CentralEncoderModule(nn.Module):
  def __init__(self, options_file):
    super(CentralEncoderModule, self).__init__()
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=options_file["model_dim"],
                                                    nhead=options_file["num_of_central_attention_heads"],
                                                    dropout=options_file["central_dropout"],
                                                    activation=options_file["central_activation_function"],
                                                    batch_first=True)

    self.encoder = nn.TransformerEncoder(self.encoder_layer, options_file["num_of_central_encoder_layers"])

  def forward(self, input):
    return self.encoder(input)


class CentralEncoder(nn.Module):
  def __init__(self, options_file):
    super(CentralEncoder, self).__init__()
    self.encoder_modules = nn.Sequential(*[CentralEncoderModule(options_file) for i in range(options_file["num_of_central_encoders"])])

  def forward(self, input):
    x = self.encoder_modules(input)

    return x


class ParticleEncoderModule(nn.Module):
  def __init__(self, options_file):
    super(ParticleEncoderModule, self).__init__()
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=options_file["model_dim"],
                                                    nhead=options_file["num_of_particle_attention_heads"],
                                                    dropout=options_file["particle_dropout"],
                                                    activation=options_file["particle_activation_function"],
                                                    batch_first=True)

    self.encoder = nn.TransformerEncoder(self.encoder_layer, options_file["num_of_particle_encoder_layers"])

  def forward(self, input):
    return self.encoder(input)


class ParticleEncoder(nn.Module):
  def __init__(self, options_file):
    super(ParticleEncoder, self).__init__()
    self.encoder_modules = nn.Sequential(*[ParticleEncoderModule(options_file) for i in range(options_file["num_of_particle_encoders"])])

  def forward(self, input):
    x = self.encoder_modules(input)

    return x


class TensorAttention(nn.Module):
    def __init__(self, options_file):
        super(TensorAttention, self).__init__()
        self.model_dim = options_file["model_dim"]
        self.weights = nn.Parameter(torch.rand(self.model_dim, self.model_dim))

    def forward(self, input):
        symmetric_weights = (self.weights + self.weights.t())/2
        self.output = torch.bmm(input, torch.bmm(symmetric_weights.unsqueeze(0).expand(input.size(0), -1, -1), input.transpose(1, 2)))

        inf = torch.tril(torch.full_like(self.output, float('-inf')), diagonal=0)
        self.output = self.output + inf
        self.output = F.softmax(self.output.view(input.size(0), -1), dim=-1).view(self.output.size())
        self.output = self.output + self.output.transpose(1, 2)

        '''
        self.mask = torch.zeros_like(self.output)
        max_values, max_indices = torch.max(self.output.view(self.output.size(0), -1), dim=-1)
        self.mask.view(self.output.size(0), -1)[torch.arange(self.mask.size(0)), max_indices] = 1
        self.mask = self.mask + self.mask.transpose(1, 2)
        '''

        return self.output
    

class Categorizer(nn.Module):
    def __init__(self, options_file):
        super(Categorizer, self).__init__()
        self.num_of_layers = options_file["num_of_categorizer_layers"]
        self.first_layer_size = 2 * options_file["num_of_jets"] * options_file["model_dim"]
        self.last_layer_size = len(options_file["event_topology"])
        self.step = (self.first_layer_size - self.last_layer_size) / (self.num_of_layers - 1)
        self.threshold = options_file["categorization_threshold"]

        # First linear layer
        self.first_layer = nn.Linear(options_file["num_of_jets"] * options_file["model_dim"], self.first_layer_size)

        # Hidden layers
        layers = []
        for i in range(self.num_of_layers - 1):
            layers.append(nn.Linear(round(self.first_layer_size - i * self.step), round(self.first_layer_size - (i + 1) * self.step)))
            if i != self.num_of_layers - 2:
              layers.append(nn.GELU())
            else:
              layers.append(nn.Sigmoid())

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
      x = input.view(input.size(0), -1)
      x = nn.functional.gelu(self.first_layer(x))
      x = self.hidden_layers(x)
      #x = x.ge(self.threshold).float()

      return x


class ParticlePass(nn.Module):
  def __init__(self, options_file):
    super(ParticlePass, self).__init__()
    self.particle_encoder = ParticleEncoder(options_file)
    self.tensor_attention = TensorAttention(options_file)

  def forward(self, input):
    self.encoder_output = self.particle_encoder(input)
    self.assignment = self.tensor_attention(self.encoder_output)

    return self.encoder_output, self.assignment


#MODELS

class TAPTE(nn.Module):
  def __init__(self, options_file):
    super(TAPTE, self).__init__()
    self.num_of_particles = len(options_file["event_topology"])
    self.initial_embedding = InitialEmbedding(options_file)
    self.central_encoder = CentralEncoder(options_file)
    self.categorizer = Categorizer(options_file)
    self.particle_passes = nn.ModuleList([ParticlePass(options_file) for i in range(self.num_of_particles)])
    self.threshold = options_file["categorization_threshold"]
    self.options_file = options_file

  def forward(self, input):
    x = self.initial_embedding(input)
    x = self.central_encoder(x)
    particles = [particle_pass(x) for particle_pass in self.particle_passes]

    self.assignments = torch.stack([particles[i][1] for i in range(self.num_of_particles)], dim=1)
    self.encodings = torch.stack([particles[i][0] for i in range(self.num_of_particles)], dim=0)

    self.categories = self.categorizer(torch.mean(self.encodings, dim=0))
    mask = self.categories.ge(self.threshold).float()

    self.assignments = self.assignments * mask.view(self.options_file["batch_size"], self.num_of_particles, 1, 1)

    return self.assignments, self.categories

class TAPTELightning(L.LightningModule):
    def __init__(self, options_file):
        super(TAPTELightning, self).__init__()
        self.model = TAPTE(options_file)
        self.loss_function = HybridSymmetricLoss(options_file)
        self.options_file = options_file

    def training_step(self, batch, batch_idx):
        inputs, assignment_labels, category_labels = batch
        assignment_preds, category_preds = self.model(inputs)
        loss = self.loss_function(assignment_preds, category_preds, assignment_labels, category_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, assignment_labels, category_labels = batch
        assignment_preds, category_preds = self.model(inputs)
        loss = self.loss_function(assignment_preds, category_preds, assignment_labels, category_labels)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, assignment_labels, category_labels = batch
        assignment_preds, category_preds = self.model(inputs)
        loss = self.loss_function(assignment_preds, category_preds, assignment_labels, category_labels)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.options_file["learning_rate"])
        return optimizer


#LOSS FUNCTION

class HybridSymmetricLoss(nn.Module):
  def __init__(self, options_file):
    super(HybridSymmetricLoss, self).__init__()
    self.loss = nn.BCELoss()
    self.assignments_loss_weight = options_file["assignments_loss_weight"]
    self.categorizer_loss_weight = options_file["categorizer_loss_weight"]

  def forward(self, assignments, category, assignments_labels, category_labels):
    device = assignments.device
    indices = torch.arange(assignments.size(1))
    min_loss = torch.ones(assignments.size(0)) * float('inf')
    perm = torch.empty(assignments.size(0), assignments.size(1))
        
    for i in permutations(indices):
        for event in range(assignments.size(0)):
            l = self.loss(assignments[event, list(i), :, :], assignments_labels[event])
            if l < min_loss[event]:
                min_loss[event] = l
                perm[event] = torch.tensor(list(i))

    batch_losses, batch_perms = min_loss.to(device), perm.long().to(device)
    permuted_category = torch.gather(category, 1, batch_perms).to(device)

    assignments_loss = torch.mean(batch_losses)
    category_loss = self.loss(permuted_category, category_labels)

    return self.assignments_loss_weight * assignments_loss + self.categorizer_loss_weight * category_loss

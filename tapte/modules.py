import numpy as np
import math
import os
import h5py
from prettytable import PrettyTable

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from itertools import permutations


#UTILITIES

def create_or_load_version_directory(base_dir, checkpoint):
    if checkpoint:
        version_dir = os.path.join(base_dir, checkpoint)
    else:
        existing_versions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        next_version = 1 if not existing_versions else max([int(version.split('_')[1]) for version in existing_versions]) + 1
        version_dir = os.path.join(base_dir, f'version_{next_version}')
        os.makedirs(version_dir, exist_ok=True)
        print(f'version_{next_version}')

    return version_dir

def print_dict_as_table(input_dict):
      table = PrettyTable(["Key", "Value"])
      for key, value in input_dict.items():
          table.add_row([key, value])
      
      print(table)

def print_h5_tree(group, indent=0):
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print("  " * indent + f"Group: {name}")
            print_h5_tree(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")


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
    def __init__(self, options_file, split='train'):
        self.file_path = options_file["dataset"]
        self.size = options_file["optional_loading_size"]
        self.split_ratios = options_file["split"]
        assert sum(self.split_ratios) == 1.0, "Split ratios should sum up to 1.0"
        assert len(self.split_ratios) == 3, "Split requires three values corresponding to training, validation and testing ratios respectfully"

        self.variables = options_file["variables"]

        train_size, val_size, test_size = self.split_ratios

        with h5py.File(self.file_path, 'r') as file:
          if self.size is None:
            all_indices = list(range(len(file['TARGETS/h1/mask'])))
          else:
            all_indices = list(range(self.size))

          if split == 'train':
              self.indices = all_indices[:int(train_size * len(all_indices))]
          elif split == 'val':
              self.indices = all_indices[int(train_size * len(all_indices)):int((train_size + val_size) * len(all_indices))]
          elif split == 'test':
              self.indices = all_indices[int((1 - test_size) * len(all_indices)):]
          else:
              raise ValueError("Invalid split parameter. Use 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
      with h5py.File(self.file_path, 'r') as self.file:
        actual_index = self.indices[index]

        # Load inputs
        input = []
        for variable in self.variables:
            input.append(self.file[f'INPUTS/Jets/{variable}'][actual_index])

        input = np.array(input, dtype=np.float32)
        input = torch.tensor(input, dtype=torch.float).permute(1, 0)

        # Load assignments
        h1_data = self.file['TARGETS']['h1']
        h2_data = self.file['TARGETS']['h2']
        h3_data = self.file['TARGETS']['h3']

        indices_h1 = [self.file['TARGETS']['h1']['b1'][actual_index], self.file['TARGETS']['h1']['b2'][actual_index]]
        indices_h2 = [self.file['TARGETS']['h2']['b1'][actual_index], self.file['TARGETS']['h2']['b2'][actual_index]]
        indices_h3 = [self.file['TARGETS']['h3']['b1'][actual_index], self.file['TARGETS']['h3']['b2'][actual_index]]

        perm_indices_h1 = list(permutations(indices_h1))
        perm_indices_h2 = list(permutations(indices_h2))
        perm_indices_h3 = list(permutations(indices_h3))

        assignment = np.zeros((3, 10, 10), dtype=np.float32)

        if -1 not in indices_h1: assignment[0, perm_indices_h1[0], perm_indices_h1[1]] = 1
        if -1 not in indices_h2: assignment[1, perm_indices_h2[0], perm_indices_h2[1]] = 1
        if -1 not in indices_h3: assignment[2, perm_indices_h3[0], perm_indices_h3[1]] = 1

        assignment = torch.tensor(assignment, dtype=torch.float)

        # Load categories
        mask_h1 = h1_data['mask'][actual_index].astype(np.float32)
        mask_h2 = h2_data['mask'][actual_index].astype(np.float32)
        mask_h3 = h3_data['mask'][actual_index].astype(np.float32)

        category = torch.tensor([mask_h1, mask_h2, mask_h3], dtype=torch.float)

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

        # Hidden layers with batch normalization
        layers = []
        for i in range(self.num_of_layers - 1):
            layers.append(nn.Linear(round(self.first_layer_size - i * self.step), round(self.first_layer_size - (i + 1) * self.step)))
            layers.append(nn.BatchNorm1d(round(self.first_layer_size - (i + 1) * self.step), track_running_stats=False))
            if i != self.num_of_layers - 2:
                layers.append(nn.GELU())

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.first_layer(x)
        x = nn.GELU()(x)
        x = self.hidden_layers(x)
        x = nn.Sigmoid()(x)

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
    self.options_file = options_file
    self.num_of_particles = len(options_file["event_topology"])
    self.initial_embedding = InitialEmbedding(options_file)
    self.central_encoder = CentralEncoder(options_file)
    self.categorizer = Categorizer(options_file)
    self.particle_passes = nn.ModuleList([ParticlePass(options_file) for i in range(self.num_of_particles)])
    self.threshold = options_file["categorization_threshold"]

  def forward(self, input):
    x = self.initial_embedding(input)
    x = self.central_encoder(x)
    particles = [particle_pass(x) for particle_pass in self.particle_passes]

    self.assignments = torch.stack([particles[i][1] for i in range(self.num_of_particles)], dim=1)
    self.encodings = torch.stack([particles[i][0] for i in range(self.num_of_particles)], dim=0)

    self.categories = self.categorizer(torch.mean(self.encodings, dim=0))

    mask = self.categories.ge(self.threshold).float()
    self.assignments = self.assignments * mask.view(self.assignments.size(0), self.num_of_particles, 1, 1)

    return self.assignments, self.categories

class TAPTELightning(L.LightningModule):
    def __init__(self, options_file, model):
        super(TAPTELightning, self).__init__()
        self.model = model
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
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.loss_function.inference = True
        inputs, assignment_labels, category_labels = batch
        assignment_preds, category_preds = self.model(inputs)
        loss = self.loss_function(assignment_preds, category_preds, assignment_labels, category_labels)
        return loss

    def on_test_epoch_end(self):
        self.loss_function.print_results()

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.options_file["learning_rate"])
        return self.optimizer


#CALLBACKS

class PrintOptionsFileCallback(L.Callback):
    def __init__(self, options_file):
      super(PrintOptionsFileCallback, self).__init__()
      self.options_file = options_file

    def on_train_start(self, trainer, pl_module):
        print("options_file content:")
        print_dict_as_table(self.options_file)


#LOSS FUNCTION

def one_hot_encode(input):
  if torch.any(input != 0):
    max_value = input.view(-1).max()
    max_indices = torch.nonzero(input == max_value)

    pick_indices = []
    for i in range(max_indices.size(1)):
      pick_indices.append(max_indices[:, i])

    one_hot_encoded = torch.zeros_like(input)
    one_hot_encoded[pick_indices] = 1

    return one_hot_encoded

  else:
    return input

def count_correct_Higgs(X, Y):

  X_set = {tuple(matrix.flatten().detach().cpu().numpy()) for matrix in X if torch.any(matrix != 0)}
  Y_set = {tuple(matrix.flatten().detach().cpu().numpy()) for matrix in Y}

  count = sum(matrix in Y_set for matrix in X_set)

  return count

class HybridSymmetricLoss(nn.Module):
  def __init__(self, options_file):
    super(HybridSymmetricLoss, self).__init__()
    self.loss = nn.BCELoss()
    self.assignments_loss_weight = options_file["assignments_loss_weight"]
    self.categorizer_loss_weight = options_file["categorizer_loss_weight"]

    self.inference = False
    self.num_of_categories = torch.zeros(len(options_file["event_topology"]) + 1)
    self.confusion_matrix = torch.zeros(2, self.num_of_categories.size(0), self.num_of_categories.size(0))

  def print_results(self):
    event_proportions = torch.zeros_like(self.num_of_categories)
    category_percentages = torch.zeros_like(self.confusion_matrix[0])
    higgs_percentages = torch.zeros_like(self.confusion_matrix[1])

    for i in range(event_proportions.size(0)):
      event_proportions[i] = round(self.num_of_categories[i].item() / torch.sum(self.num_of_categories).item(), 4)

    event_proportions = event_proportions.numpy()
    event_proportions_results = PrettyTable()
    event_proportions_results.field_names = ["10 Jets", "0h", "1h", "2h", "3h"]
    event_proportions_results.add_rows(
        [
            ["Proportions", event_proportions[0], event_proportions[1], event_proportions[2], event_proportions[3]]
        ]
    )
    
    print("Event proportions:")
    print(event_proportions_results)

    for i in range(category_percentages.size(0)):
      for j in range(category_percentages.size(1)):
        category_percentages[i][j] = round(self.confusion_matrix[0][i][j].item() / torch.sum(self.confusion_matrix[0][i]).item(), 4)
        if self.confusion_matrix[0][i][j].item() * i != 0:
          higgs_percentages[i][j] = round(self.confusion_matrix[1][i][j].item() / (torch.sum(self.confusion_matrix[0][i]).item() * i), 4)
        else:
          higgs_percentages[i][j] = math.nan

    category_percentages = category_percentages.numpy()
    categories_results = PrettyTable()
    categories_results.field_names = ["10 Jets", "0h", "1h", "2h", "3h"]
    categories_results.add_rows(
        [
            ["0h", category_percentages[0][0], category_percentages[0][1], category_percentages[0][2], category_percentages[0][3]],
            ["1h", category_percentages[1][0], category_percentages[1][1], category_percentages[1][2], category_percentages[1][3]],
            ["2h", category_percentages[2][0], category_percentages[2][1], category_percentages[2][2], category_percentages[2][3]],
            ["3h", category_percentages[3][0], category_percentages[3][1], category_percentages[3][2], category_percentages[3][3]],
        ]
    )

    print("Categorization results:")
    print(categories_results)

    higgs_percentages = higgs_percentages.numpy()
    higgs_results = PrettyTable()
    higgs_results.field_names = ["10 Jets", "0h", "1h", "2h", "3h"]
    higgs_results.add_rows(
        [
            ["0h", higgs_percentages[0][0], higgs_percentages[0][1], higgs_percentages[0][2], higgs_percentages[0][3]],
            ["1h", higgs_percentages[1][0], higgs_percentages[1][1], higgs_percentages[1][2], higgs_percentages[1][3]],
            ["2h", higgs_percentages[2][0], higgs_percentages[2][1], higgs_percentages[2][2], higgs_percentages[2][3]],
            ["3h", higgs_percentages[3][0], higgs_percentages[3][1], higgs_percentages[3][2], higgs_percentages[3][3]],
        ]
    )

    print("Assignment results:")
    print(higgs_results)

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

    permuted_assignments = assignments.clone()
    for i in range(permuted_assignments.size(0)):
      permuted_matrices = permuted_assignments[i, batch_perms[i]]

      # Update the original tensor with permuted matrices
      permuted_assignments[i] = permuted_matrices

    permuted_category = torch.gather(category, 1, batch_perms).to(device)

    assignments_loss = torch.mean(batch_losses)
    category_loss = self.loss(permuted_category, category_labels)

    #Update results:

    if self.inference == True:

      for i in range(permuted_assignments.size(0)):
        for j in range(permuted_assignments.size(1)):
          permuted_assignments[i][j] = one_hot_encode(permuted_assignments[i][j])

      permuted_category = permuted_category.ge(0.5).float()

      labels = torch.sum(category_labels, dim=1).long()
      predicted_labels = torch.sum(permuted_category, dim=1).long()

      for k in range(labels.size(0)):
        self.num_of_categories[labels[k]] += 1
        self.confusion_matrix[0, labels[k], predicted_labels[k]] += 1
        self.confusion_matrix[1, labels[k], predicted_labels[k]] += count_correct_Higgs(permuted_assignments[k], assignments_labels[k])

    return self.assignments_loss_weight * assignments_loss + self.categorizer_loss_weight * category_loss

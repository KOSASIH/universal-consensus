# nta.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NTADataset(Dataset):
    def __init__(self, network_data):
        self.data = network_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class NTA(nn.Module):
    def __init__(self):
        super(NTA, self).__init__()
        self.fc1 = nn.Linear(1024, 128)  # Input layer (1024) -> Hidden layer (128)
        self.fc2 = nn.Linear(128, 64)   # Hidden layer (128) -> Output layer (64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)
        return x

def train_nta(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def analyze_traffic(model, device, data):
    model.eval()
    output = model(data.to(device))
    return output.detach().cpu().numpy()

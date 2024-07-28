# train_neural_network.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_neural_network(model, data, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data['features'])
        loss = criterion(outputs, data['labels'])
        loss.backward()
        optimizer.step()
        
    return model

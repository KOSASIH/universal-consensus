import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralNetworkModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = NeuralNetwork(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train):
        self.model.train()
        for epoch in range(10):
            for x, y in zip(X_train, y_train):
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.long)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x in X_test:
                x = torch.tensor(x, dtype=torch.float32)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())
        return predictions

import torch
import torch.nn as nn
import torch.optim as optim

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvolutionalNeuralNetworkModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = ConvolutionalNeuralNetwork(input_dim, hidden_dim, output_dim)
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

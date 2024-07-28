import torch
import torch.nn as nn
import torch.optim as optim

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecurrentNeuralNetwork, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class RecurrentNeuralNetworkModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = RecurrentNeuralNetwork(input_dim, hidden_dim, output_dim)
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

    def evaluate(self, X_test, y_test):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in zip(X_test, y_test):
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.long)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
        accuracy = correct / len(y_test)
        return accuracy

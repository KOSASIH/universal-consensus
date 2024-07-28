import torch
import torch.nn as nn
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        output = self.fc(decoder_output)
        return output

class TransformerModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = Transformer(input_dim, hidden_dim, output_dim)
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

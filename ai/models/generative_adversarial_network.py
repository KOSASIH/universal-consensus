import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GenerativeAdversarialNetworkModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.generator = Generator(input_dim, hidden_dim, output_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim, output_dim)
        self.criterion = nn.BCELoss()
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, X_train):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(10):
            for x in X_train:
                x = torch.tensor(x, dtype=torch.float32)
                z = torch.randn(x.size(0), x.size(1))
                fake_x = self.generator(z)
                real_output = self.discriminator(x)
                fake_output = self.discriminator(fake_x)
                lossD = self.criterion(real_output, torch.ones_like(real_output)) + self.criterion(fake_output, torch.zeros_like(fake_output))
                lossG = self.criterion(fake_output, torch.ones_like(fake_output))
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.optimizerG.zero_grad()
                lossG.backward()
                self.optimizerG.step()

    def generate(self, z):
        self.generator.eval()
        with torch.no_grad():
            fake_x = self.generator(z)
        return fake_x

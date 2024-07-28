import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        x_recon = self.fc3(z)
        return x_recon

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class VariationalAutoencoderModel:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.model = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train):
        self.model.train()
        for epoch in range(10):
            for x in X_train:
                x = torch.tensor(x, dtype=torch.float32)
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x)
                loss = self.loss_function(x, x_recon, mu, logvar)
                loss.backward()
                self.optimizer.step()

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = self.criterion(x_recon, x)
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar)
        loss = recon_loss + kl_loss
        return loss

    def generate(self, z):
        self.model.eval()
        with torch.no_grad():
            x_recon = self.model.decoder(z)
        return x_recon

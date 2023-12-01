import torch
import torch.nn as nn


class vae(nn.Module):
    # We want the bottleneck to be paramaterized by mean and logvariance
    def __init__(self):
        super().__init__()
        self.encoder = encoder(latentdim=20)
        self.decoder = decoder(latentdim=20)

    def sample(self, mean, logvar):
        variance = torch.exp(logvar)
        std = torch.sqrt(variance)
        sample = torch.normal(mean, std)
        return sample

    def forward(self, x):
        x = x.view(-1, 784)
        mean, logvar = self.encoder(x)
        vector = self.sample(mean, logvar)
        output = self.decoder(vector)
        return output, mean, logvar


class encoder(nn.Module):
    def __init__(self, in_channels=1, latentdim=2):
        # returns tuple (mean, logvar)
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(),
            nn.ReLU(),
        )
        self.linearMean = nn.Linear(64, latentdim)
        self.linearLogvar = nn.Linear(64, latentdim)

    def forward(self, x):
        x = self.encoder(x)
        mean = self.linearMean(x)
        logvar = self.linearLogvar(x)
        return mean, logvar

class decoder(nn.Module):
    def __init__(self, latentdim=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latentdim, 32),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(),
            nn.ReLU(),

            nn.Linear(64, 784)
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x

def conv_vae():
    model = vae()
    return model
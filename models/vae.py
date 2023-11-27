import torch
import torch.nn as nn


class vae(nn.Module):
    # We want the bottleneck to be paramaterized by mean and logvariance
    def __init__(self):
        super().__init__()
        self.encoder = encoder(latentdim=6)
        self.decoder = decoder()

    def sample(self, mean, logvar):
        variance = torch.exp(logvar)
        std = torch.sqrt(variance)
        sample = torch.normal(mean, std)
        return sample

    def forward(self, x):
        mean, logvar = self.encoder(x)
        vector = self.sample(mean, logvar)
        output = self.decoder(vector)
        return output


class encoder(nn.Module):
    def __init__(self, in_channels=1, latentdim=2):
        # returns tuple (mean, logvar)
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linearMean = nn.Linear(64, latentdim)
        self.linearLogvar = nn.Linear(64, latentdim)

    def forward(self, x):
        x = self.encoder(x)
        mean = self.linearMean(x)
        logvar = self.linearLogvar(x)
        return mean, logvar

class decoder(nn.Module):
    def __init__(self, latentdim):
        super().__init__()
        self.process = nn.Linear(latentdim, 49)
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.process(x)
        x = x.view(-1, 1, 7, 7)
        x = self.decoder(x)
        return x

def conv_vae():
    model = vae()
    return model
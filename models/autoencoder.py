import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.bottleneck = nn.Linear(2, 196)
        self.decoder = decoder()

    def forward(self, x):
        bottleneck = self.encoder(x)
        # Input in R2 -> 128 neurons to (4, 7, 7)
        x = bottleneck.view(-1, 4, 7, 7)
        x = self.decoder(x)
        return x
    
class decoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)

class encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(64, 2),
            nn.BatchNorm2d(2),
            nn.Sigmoid(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)
    
def conv_autoencoder():
    model = autoencoder()
    return model
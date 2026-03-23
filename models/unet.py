import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up_block = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t=None):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)

        x4 = self.up(x3)
        x_cat = torch.cat([x4, x1], dim=1)

        out = self.up_block(x_cat)
        out = self.out(out)

        return out

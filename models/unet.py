import torch
import torch.nn as nn

import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1)
        return self.net(t)


class SimpleUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        # self.time_mlp = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

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

        self.time_to_64 = nn.Linear(time_dim, 64)
        self.time_to_128 = nn.Linear(time_dim, 128)


    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        t1 = self.time_to_64(t_emb).unsqueeze(-1).unsqueeze(-1)
        t2 = self.time_to_128(t_emb).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x)
        x1 = x1 + t1

        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x3 = x3 + t2

        x4 = self.up(x3)
        x_cat = torch.cat([x4, x1], dim=1)

        out = self.up_block(x_cat)
        out = self.out(out)

        return out
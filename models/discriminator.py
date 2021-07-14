import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # Input: (self.nc) x 128 x 128
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (self.ndf) x 64 x 64

            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (self.ndf * 2) x 32 x 32

            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (self.ndf * 4) x 16 x 16

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (self.ndf * 8) x 8 x 8

            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (self.ndf * 16) x 4 x 4

            nn.Conv2d(self.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, inputs):
        return self.main(inputs)

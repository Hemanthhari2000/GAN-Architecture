import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # Output: (self.ngf*16) x 4 x 4

            nn.ConvTranspose2d(self.ngf * 16, self.ngf * \
                               8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # Output: (self.ngf*8) x 8 x 8

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # Output: (self.ngf*4) x 16 x 16

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # Output: (self.ngf*2) x 32 x 32

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # Output: (self.ngf) x 64 x 64

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output: (self.nc) x 128 x 128
        )

    def forward(self, inputs):
        return self.main(inputs)

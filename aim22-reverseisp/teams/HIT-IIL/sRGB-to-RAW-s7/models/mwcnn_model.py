import torch
import networks as N
import torch.nn as nn
import math
import torch.optim as optim

class MWRCAN(nn.Module):
    def __init__(self):
        super(MWRCAN, self).__init__()
        c1 = 64
        c2 = 128
        c3 = 128
        n_b = 20
        self.head = N.DWTForward()

        self.down1 = N.seq(
            nn.Conv2d(4 * 4, c1, 3, 1, 1),
            nn.PReLU(),
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.down2 = N.seq(
            N.DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
              N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b)
        )

        self.down3 = N.seq(
            N.DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = N.seq(
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b),
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b)
        )
        
        self.up1 = N.seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse()
        )

        self.up2 = N.seq(
            N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse()
        )

        self.up3 = N.seq(
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, 16, 3, 1, 1)
        )

        self.tail = N.seq(
            N.DWTInverse(),
            nn.Conv2d(4, 12, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x, c=None):
        c0 = x
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = False

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
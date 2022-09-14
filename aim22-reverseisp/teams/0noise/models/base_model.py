import torch.nn as nn
from models import activation


class RB(nn.Module):
    """Residual Block"""
    def __init__(self, bc, act="silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(bc, bc * 2, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.act1 = activation(act)
        self.conv2 = nn.Conv2d(bc * 2, bc, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.act2 = activation(act)

    def forward(self, x):
        xx = self.act1(self.conv1(x))
        xx = self.conv2(xx) + x
        return self.act2(xx)

class CSTB(nn.Module):
    """Color Shift Transformation Block"""
    def __init__(self, ic, bc, act="silu"):
        super().__init__()
        self.block = nn.Sequential(
            CB(ic, bc, ks=3, act=act),
            CB(bc, bc, ks=3, act=act),
            CB(bc, bc, ks=3, act=act),
            nn.Conv2d(bc , bc, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.block(x)


class CB(nn.Module):
    """Convolutional Block"""
    def __init__(self, ic, oc, act="silu", ks=3):
        super().__init__()
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, bias=True)
        else:
            padding = ks // 2
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=padding, padding_mode="reflect", bias=True)
        self.act = activation(act)

    def forward(self, x):
        return self.act(self.conv(x))


class TMB(nn.Module):
    """Tone Mapping Block"""
    def __init__(self, ic, bc, act="silu"):
        super().__init__()
        self.block = nn.Sequential(
            CB(ic, bc, ks=1, act=act),
            CB(bc, bc, ks=1, act=act),
            CB(bc, bc, ks=1, act=act),
            nn.Conv2d(bc , bc, kernel_size=1, bias=True)
        )


    def forward(self, x):
        return self.block(x)

class LSB(nn.Module):
    """Lens Shading Block"""
    def __init__(self, ic, bc, act="silu"):
        super().__init__()
        self.block = nn.Sequential(
            CB(ic, bc, ks=3, act=act),
            CB(bc, bc, ks=3, act=act),
            CB(bc, bc, ks=3, act="sigmoid"),
        )

    def forward(self, x):
        return self.block(x)


class Base(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

        ic = 3
        bc = opt.base_channels

        self.p = CB(ic, bc, ks=1)
        self.tm = TMB(bc, bc)
        self.r1 = RB(bc)
        self.cstb = CSTB(bc, bc)
        self.r2 = RB(bc)
        self.lsb = LSB(bc, bc)
        self.cout = nn.Conv2d(bc, 4, kernel_size=2, padding=0, stride=2, bias=True)
        self.aout = activation("relu")


    def forward(self, x):


        xp = self.p(x)
        x_ = xp * self.tm(xp)
        x_ = self.r1(x_)
        x_ = x_ * self.cstb(xp)
        x_ = self.r2(x_)
        x_ = x_ * self.lsb(xp)
        out = self.aout(self.cout(x_))
        return out

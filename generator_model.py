import torch
import torch.nn as nn

#生成器

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, encode=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if encode
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.encode = encode

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    ##11


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_encode = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.encode1 = Block(features, features * 2, encode=True, act="leaky", use_dropout=False)
        self.encode2 = Block(
            features * 2, features * 4, encode=True, act="leaky", use_dropout=False
        )
        self.encode3 = Block(
            features * 4, features * 8, encode=True, act="leaky", use_dropout=False
        )
        self.encode4 = Block(
            features * 8, features * 8, encode=True, act="leaky", use_dropout=False
        )
        self.encode5 = Block(
            features * 8, features * 8, encode=True, act="leaky", use_dropout=False
        )
        self.encode6 = Block(
            features * 8, features * 8, encode=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.decode1 = Block(features * 8, features * 8, encode=False, act="relu", use_dropout=True)
        self.decode2 = Block(
            features * 8 * 2, features * 8, encode=False, act="relu", use_dropout=True
        )
        self.decode3 = Block(
            features * 8 * 2, features * 8, encode=False, act="relu", use_dropout=True
        )
        self.decode4 = Block(
            features * 8 * 2, features * 8, encode=False, act="relu", use_dropout=False
        )
        self.decode5 = Block(
            features * 8 * 2, features * 4, encode=False, act="relu", use_dropout=False
        )
        self.decode6 = Block(
            features * 4 * 2, features * 2, encode=False, act="relu", use_dropout=False
        )
        self.decode7 = Block(features * 2 * 2, features, encode=False, act="relu", use_dropout=False)
        self.final_decode = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_encode(x)
        d2 = self.encode1(d1)
        d3 = self.encode2(d2)
        d4 = self.encode3(d3)
        d5 = self.encode4(d4)
        d6 = self.encode5(d5)
        d7 = self.encode6(d6)
        bottleneck = self.bottleneck(d7)
        decode1 = self.decode1(bottleneck)
        decode2 = self.decode2(torch.cat([decode1, d7], 1))
        decode3 = self.decode3(torch.cat([decode2, d6], 1))
        decode4 = self.decode4(torch.cat([decode3, d5], 1))
        decode5 = self.decode5(torch.cat([decode4, d4], 1))
        decode6 = self.decode6(torch.cat([decode5, d3], 1))
        decode7 = self.decode7(torch.cat([decode6, d2], 1))
        return self.final_decode(torch.cat([decode7, d1], 1))

import torch

from typing import Optional

class ResidualBlock(torch.nn.Module):
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    downsample : Optional
    relu : torch.nn.ReLU
    out_channels : int
    def __init__(self, in_channels:int, out_channels:int, stride:int=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x:torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNet(torch.nn.Module):
    conv : torch.nn.Sequential
    layer0 : torch.nn.Sequential
    layer1 : torch.nn.Sequential
    layer2 : torch.nn.Sequential
    layer3 : torch.nn.Sequential
    pool : torch.nn.AvgPool2d
    linear : torch.nn.Linear
    channels : int
    def __init__(self, layers:list[int], in_channels:int, classes:int, block:torch.nn.Module=ResidualBlock, channels:int=64):
        super().__init__()
        self.channels = channels
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer0 = self._construct_layer(block, channels, layers[0], stride=1)
        self.layer1 = self._construct_layer(block, channels*2, layers[1], stride=2)
        self.layer2 = self._construct_layer(block, channels*4, layers[2], stride=2)
        self.layer3 = self._construct_layer(block, channels*8, layers[3], stride=2)
        self.pool = torch.nn.AvgPool2d(7, stride=1)
        self.linear = torch.nn.Linear(channels*8, classes)

    def _construct_layer(self, block:torch.nn.Module, planes:int, blocks:int, stride:int):
        downsample = None
        if stride != 1 or self.channels != planes:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.channels, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes)
            )
        layers = [block(self.channels, planes, stride, downsample)]
        self.channels = planes
        for i in range(1, blocks):
            layers.append(block(self.channels, planes))
        return torch.nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        x = self.layer0(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)

        x = self.pool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear(x)
        print(x.shape)

        return x

def ResNet14(channels:int=1, classes:int=10):
    return ResNet([3,3,3,3], channels, classes)
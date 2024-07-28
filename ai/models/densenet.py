import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1 = self._make_dense_layer(64, 128, 6)
        self.transition1 = self._make_transition_layer(128, 256)
        self.dense2 = self._make_dense_layer(256, 512, 12)
        self.transition2 = self._make_transition_layer(512, 1024)
        self.dense3 = self._make_dense_layer(1024, 2048, 24)
        self.transition3 = self._make_transition_layer(2048, 4096)
        self.fc = nn.Linear(4096, num_classes)

    def _make_dense_layer(self, inplanes, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(DenseBlock(inplanes, planes))
            inplanes += planes
        return nn.Sequential(*layers)

    def _make_transition_layer(self, inplanes, planes):
        return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(),
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out =

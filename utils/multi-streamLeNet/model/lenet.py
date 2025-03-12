import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, ModuleList, LogSoftmax
from torch import flatten

class LeNet(Module):
    def __init__(self, numChannels, feature_size=500):
        super(LeNet, self).__init__()

        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = Linear(in_features=1250, out_features=feature_size)
        self.relu3 = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        return x


class MultiStreamCNN(Module):
    def __init__(self, numChannels, num_classes, feature_size=500):
        super(MultiStreamCNN, self).__init__()
        
        self.streams = ModuleList([
            LeNet(numChannels, feature_size) for _ in range(9)
        ])
        
        self.classifier = Linear(feature_size * 9, num_classes)
        
    def forward(self, histograms):
        
        features = []
        for i, histogram in enumerate(histograms):
            feature = self.streams[i](histogram)
            features.append(feature)
        
        combined_features = torch.cat(features, dim=1)
        
        output = self.classifier(combined_features)
        
        return output
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()

        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride = (2,2))

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2= MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # self.fc1 = Linear(in_features = 800, out_features=500)
        self.fc1 = Linear(in_features = 1250, out_features=500)

        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=500, out_features=classes)
        # self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        output = self.fc2(x)
        # output = self.logSoftmax(x)
        # return the output predictions
        return output


class MultiStreamCNN(nn.Module):
    def __init__(self, numChannels, num_classes):
        super(MultiStreamCNN, self).__init__()
        
        # Histogram stream
        self.hist_cnns = nn.ModuleList([LeNet(numChannels, num_classes) for _ in range(9)])
        
        # Combine streams
        self.fc = nn.Linear(in_features=num_classes*9, out_features=num_classes)
        
    def forward(self, histograms):

        # Stack histograms along the channel dimension
        histograms_tensor = torch.stack(histograms, dim=1)
        
        # Process histogram streams
        hist_features = []
        for i in range(9):
            hist_feature = self.hist_cnns[i](histograms_tensor[:, i])  
            hist_feature = hist_feature.view(hist_feature.size(0), -1)
            
            hist_features.append(hist_feature)
        
        # Combine histogram features
        hist_features = torch.cat(hist_features, dim=1)
        
        # Classification
        output = self.fc(hist_features)
        return output
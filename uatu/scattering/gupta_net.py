import torch.nn as nn
#from torch.nn import functional as F

from .resnet import conv3x3

class BasicBlock(nn.Module):
    def __init__(self, planes,  stride=1, p_dropout = 0.0):
        super(BasicBlock, self).__init__()
        assert len(planes) == 3
        self.planes = planes

        if p_dropout == 0.0:
            dropout = lambda x: x
        else:
            dropout = nn.Dropout2d(p_dropout)

        self.dropout = dropout
        self.conv1 = conv3x3(planes[0], planes[1], stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes[1], planes[2], stride)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.AvgPool2d(2, 2) 
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        #out = self.bn2(out)
        out = self.relu(out)
        out = self.downsample(out) 
        return out 

# TODO subclass of this that can except scattering 
class GuptaNet(nn.Module):
    def __init__(self, in_channels,p_dropout= 0.0,  J= 0, depth = [32, 64, 128, 128, 128, 128] ):
        super(GuptaNet, self).__init__()
        self.K = in_channels
        self.input_size = int(256/(2**J))

        self.layers = []
        assert len(depth)%2 ==0, "Must have even number of filters"
        self.depth = int(len(depth)/2)
        self.n_filters = depth

        self.n_filters.insert(0, in_channels) # add in channels to it

        for i in range(self.depth):
            self.layers.append(BasicBlock(self.n_filters[2*i:2*i+3], p_dropout=p_dropout))
            setattr(self, "layer_%d"%i, self.layers[-1])

        final_imsize = int(self.input_size/(2**self.depth)) # each block downsamples by 2

        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear((final_imsize**2)*self.n_filters[-1], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), self.K, self.input_size, self.input_size)

        for l in self.layers:
            x = l(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x= self.relu(x)
        x = self.fc2(x)
        x= self.relu(x)
        x = self.fc3(x)

        return x

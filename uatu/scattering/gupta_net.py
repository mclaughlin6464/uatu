import torch.nn as nn
#from torch.nn import functional as F

from .resnet import conv3x3, convNxN, shuffle

class BasicBlock(nn.Module):
    def __init__(self, planes,  stride=1, p_dropout = 0.0):
        super(BasicBlock, self).__init__()
        assert len(planes) == 3
        self.planes = planes

        if p_dropout == 0.0:
            dropout = lambda x: x
        else:
            dropout = nn.Dropout2d(p_dropout)

        self.bn = nn.BatchNorm2d(planes[0])
        self.dropout = dropout
        self.conv1 = conv3x3(planes[0], planes[1], stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes[1], planes[2], stride)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.AvgPool2d(2, 2) 
        self.stride = stride

    def forward(self, x):
        x = self.bn(x)
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

class ShuffleBlock(BasicBlock):

    def forward(self,x):
        return super().forward(shuffle(x))

class GuptaBlock(nn.Module):
    def __init__(self, planes, stride=1, p_dropout=0.0):
        super(GuptaBlock, self).__init__()
        assert len(planes) == 2
        self.planes = planes

        if p_dropout == 0.0:
            dropout = lambda x: x
        else:
            dropout = nn.Dropout2d(p_dropout)

        self.bn = nn.BatchNorm2d(planes[0])
        self.dropout = dropout
        self.conv1 = conv3x3(planes[0], planes[1], stride)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.AvgPool2d(2, 2)
        self.stride = stride

    def forward(self, x):
        x = self.bn(x)
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)
        #out = self.conv2(out)
        #out = self.dropout(out)
        # out = self.bn2(out)
        #out = self.relu(out)
        #out = self.downsample(out)
        return out

class FluriBlock(nn.Module):
    def __init__(self, planes, stride=2, N=3):
        super(FluriBlock, self).__init__()
        assert len(planes) == 2
        self.planes = planes

        self.conv = convNxN(N, planes[0], planes[1], stride)
        self.relu = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        #x = self.bn(x)
        out = self.conv(x)
        # out = self.bn1(out)
        #out = self.dropout(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.dropout(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.downsample(out)
        return out

        # TODO subclass of this that can except scattering
class GuptaNet(nn.Module):
    def __init__(self, in_channels,p_dropout= 0.0,  J= 0, depth = [4,12], shuffle_layers=0 ):
        super(GuptaNet, self).__init__()
        self.K = in_channels
        self.input_size = int(256/(2**J))

        self.layers = []
        #assert len(depth)%2 ==0, "Must have even number of filters"
        self.depth = int(len(depth))
        self.n_filters = depth

        self.n_filters.insert(0, in_channels) # add in channels to it

        for i in range(self.depth):
            block = GuptaBlock
            if i == self.depth-shuffle_layers:
                print('Appending shuffle block')
                raise NotImplementedError
                block = ShuffleBlock # append a shuffle block to the end

            self.layers.append(block(self.n_filters[i:i+2], p_dropout=p_dropout))
            setattr(self, "layer_%d"%i, self.layers[-1])

        self.downsample = nn.AvgPool2d(2, 2)
        final_imsize = int(self.input_size/(2**(self.depth-1))) # each block downsamples by 2
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(final_imsize, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = x.view(x.size(0), self.K, self.input_size, self.input_size)

        for i, l in enumerate(self.layers):
            x = l(x)
            if i!= len(self.layers)-1:
                x = self.downsample(x)

        #x = x.transpose(1,3).contiguous()
        x = x.view(x.size(0), -1)
        #return x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)


        return x

class FluriNet(nn.Module):
    def __init__(self, in_channels,p_dropout= 0.0,  J= 0, depth = [16,32,64,128,256], 
    filter_sizes = [7,5,5,3,3], shuffle_layers=0 ):
        super(FluriNet, self).__init__()

        self.K = in_channels
        self.input_size = int(256 / (2 ** J))

        self.layers = []
        # assert len(depth)%2 ==0, "Must have even number of filters"
        self.depth = int(len(depth))
        self.n_filters = depth

        self.n_filters.insert(0, in_channels)  # add in channels to it

        if type(filter_sizes) is int:
            self.filter_sizes = [filter_sizes for i in range(self.depth)]
        else:
            assert len(filter_sizes) == self.depth
            self.filter_sizes = filter_sizes

        for i, f in enumerate(self.filter_sizes):
            block = FluriBlock
            if i == self.depth - shuffle_layers:
                print('Appending shuffle block')
                raise NotImplementedError
                block = ShuffleBlock  # append a shuffle block to the end

            self.layers.append(block(self.n_filters[i:i+2], N=f))
            setattr(self, "layer_%d" % i, self.layers[-1])

        final_imsize = int(self.input_size / (2 ** (self.depth)))  # each block downsamples by 2
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.n_filters[-1]*(final_imsize**2), 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), self.K, self.input_size, self.input_size)

        for i, l in enumerate(self.layers):
            x = l(x)

        #x = x.transpose(1,3).contiguous()
        x = x.view(x.size(0), -1)
        #return x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

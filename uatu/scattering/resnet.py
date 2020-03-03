import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def shuffle(x):
    orig_size = x.size()
    x = x.view(x.size(0), -1)
    # shuffle the same over batch, allegedly not a problem
    # TODO this should not shuffle across channels, but that doesn't seem to matter? 
    rand_idxs = torch.randperm(x.size(1), requires_grad=False)
    x = x[:, rand_idxs]
    return x.view(orig_size)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ShuffleBlock(BasicBlock):

    def forward(self, x):
        residual = shuffle(x) # important, don't want spatial information sneaking in from lower in the network!
        x = shuffle(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = shuffle(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Scattering2dResNet(nn.Module):
    def __init__(self, in_channels,J,  k=2, n=4, depth = [32, 64], shuffle_layers=0 ):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.input_size = int(256/(2**J))

        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
        self.layers = []
        self.depth = depth-1 if type(depth) is int else len(depth)
        self.n_filters = [16 for i in range(depth)] if type(depth) is int else depth

        for i, nf in enumerate(self.n_filters):
            block = BasicBlock
            if i == len(self.n_filters)-shuffle_layers-1:
                block = ShuffleBlock # append a shuffle block to the end

            self.layers.append(self._make_layer(block, nf * k, n))
            setattr(self, "layer_%d"%i, self.layers[-1])

        #self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, 2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO function of J
        x = x.view(x.size(0), self.K, self.input_size, self.input_size)
        x = self.init_conv(x)

        for l in self.layers:
            x = l(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepResnet(nn.Module):
    def __init__(self,input_size=256,  init_downsample_factor=4,\
                 in_channels=1, n_subplanes=2, n_sublocks=4,\
                 depth = [16, 32, 64, 64, 64, 64, 64], shuffle_layers=0):
        super(DeepResnet, self).__init__()
        self.inplanes = 16 * n_subplanes
        self.ichannels = 16 * n_subplanes
        self.K = in_channels
        self.input_size = input_size# int(256/(2**J))
        self.downsample_size = int(self.input_size/init_downsample_factor)

        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(self.downsample_size)
        )
        self.layers = []
        #self.depth = depth-1 if type(depth) is int else len(depth)
        self.n_filters = [16 for i in range(depth)] if type(depth) is int else depth

        for i, nf in enumerate(self.n_filters):
            block = BasicBlock
            if i == len(self.n_filters) - shuffle_layers:
                block = ShuffleBlock  # append a shuffle block to the end
            self.layers.append(self._make_layer(block, nf * n_subplanes, n_sublocks))
            setattr(self, "layer_%d"%i, self.layers[-1])

        #self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(4)

        #self.fc = nn.Linear(64 * n_subplanes * 4, 2)
        self.fc = nn.Linear(n_subplanes * 16*self.n_filters[-1], 2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO function of J
        x = x.view(x.size(0), self.K, self.input_size, self.input_size)
        x = self.init_conv(x)

        for l in self.layers:
            x = l(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

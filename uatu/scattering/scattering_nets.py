import torch.nn as nn
import torch
from kymatio import Scattering2D

class Scattering_Net(nn.Module):
    """
    Network that computes the scattering of an input image, then collapses those into averaged spectra, and
    adds a few learned layers after that.
    """
    def __init__(self,J, depth = [32, 16] ):
        super(Scattering_Net, self).__init__()

        self.K = 1 # in_channels
        self.J = J
        self.L = 4 # fix # of rotations
        self.input_size = int(256)

        self.scattering = Scattering2D(J=J, shape=(self.input_size, self.input_size),\
                                       max_order=2, L = self.L)#.to(device)

        self.scattering_output_size =J+J*J # could remove informationless ones


        self.relu = nn.LeakyReLU(inplace=True)
        depth.append(2)
        self.layers = [nn.Linear(self.scattering_output_size, depth[0])]
        setattr(self, "layer_%d" %0, self.layers[-1])

        for i, d in enumerate(depth[1:]):
            self.layers.append(nn.Linear(depth[i-1], d))
            setattr(self, "layer_%d"%(i+1), self.layers[-1])

    def get_compressed_scattering(self,x):

        S = self.scattering(x).mean((2, 3), keepdim=True)  # .to('cpu').numpy()
        ls0, ls1, ls2 = 1, self.J, self.J*self.J
        #s0 = S[:, :ls0].squeeze().unsqueeze(1)#.to(device)
        s1 = S[:, ls0:ls0 + ls1].reshape((-1, self.J, self.L)).mean((2,), keepdim=True).squeeze()  # .unsqueeze(1)
        s2 = S[:, ls0 + ls1:].reshape((-1, self.J, self.L, self.J, self.L)).mean((2, 4), keepdim=True).squeeze().reshape( \
            (s1.shape[0], self.J * self.J))  # .unsqueeze(1)
        # print(s0.shape, s1.shape, s2.shape)
        return torch.cat([s1, s2], dim=1)

    def forward(self, x):
        x = x.view(x.size(0),self.K, self.input_size, self.input_size)

        if self.training:
            with torch.no_grad: # don't train through the scattering stuff
                x = self.get_compressed_scattering(x)
        else:
            x = self.get_compressed_scattering(x)

        for l in self.layers[:-1]:
            x = l(x)
            x = self.relu(x)

        return self.layers[-1](x)

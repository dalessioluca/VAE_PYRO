# Classes
import torch
import collections
import torch.nn.functional as F


class Encoder_CONV(torch.nn.Module):
    """ x -> z, taken from https://www.kaggle.com/ethanwharris/fashion-mnist-vae-with-pytorch-and-torchbearer """
    
    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.dim_z  = params['DIM_Z']
        assert self.width == self.height == 28

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.ch, 32, 4, 1, 2),   # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )
        
        self.linear = torch.nn.Linear(64 * 7 * 7, 2*self.dim_z)
        


    def forward(self,x):
        assert len(x.shape) == 4 
        batch_size = x.shape[0]
        
        x1 = self.conv(x).view(batch_size,-1)
        z = self.linear(x1)        
        z_mu  = z[...,:self.dim_z]
        z_std = F.softplus(z[...,self.dim_z:])
        return collections.namedtuple("z", "z_mu z_std")._make([z_mu,z_std])

    
class Decoder_CONV(torch.nn.Module):
    """ z -> x, taken from https://www.kaggle.com/ethanwharris/fashion-mnist-vae-with-pytorch-and-torchbearer """

    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.dim_z  = params['DIM_Z']
        assert self.width == self.height == 28
        
        self.upsample = torch.nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 2*self.ch, 4, 1, 2)   # B, 2*ch, 28, 28
        )

    def forward(self,z):
        assert len(z.shape) == 2 
        batch_size = z.shape[0]
        x1 = self.upsample(z).view(batch_size,64,7,7)
        tmp_mu, tmp_std  = torch.split(self.decoder(x1),self.ch,dim=1)
        x_mu  = torch.sigmoid(tmp_mu)
        x_std = F.softplus(tmp_std)
        return collections.namedtuple("x", "x_mu x_std")._make([x_mu.view(batch_size,-1),x_std.view(batch_size,-1)])

#------------------------------------
#------------------------------------
#------------------------------------
#------------------------------------
#------------------------------------

class Encoder_MLP(torch.nn.Module):
    """ x -> z """
    
    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.dim_z  = params['DIM_Z']
        self.dim_h1 = params['DIM_HIDDEN_1']
        self.dim_h2 = params['DIM_HIDDEN_2']
        self.dim_x  = self.ch*self.width*self.height
       
        self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_x, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_z)
            )

    def forward(self,x):
        assert len(x.shape) == 4 
        batch_size = x.shape[0]
        
        z = self.compute_z(x.view(batch_size,-1))
        z_mu  = z[...,:self.dim_z]
        z_std = F.softplus(z[...,self.dim_z:])
        
        return collections.namedtuple("z", "z_mu z_std")._make([z_mu,z_std])

    
class Decoder_MLP(torch.nn.Module):
    """ z -> x """
    
    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.dim_z  = params['DIM_Z']
        self.dim_h1 = params['DIM_HIDDEN_1']
        self.dim_h2 = params['DIM_HIDDEN_2']
        self.dim_x  = self.ch*self.width*self.height
        
        self.compute_x = torch.nn.Sequential(
                torch.nn.Linear(self.dim_z, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, 2*self.dim_x)
            )
    
    def forward(self,z):
        assert len(z.shape) == 2 
        batch_size = z.shape[0]
        x = self.compute_x(z)
        
        x_mu  = x[...,:self.dim_x]
        x_std = F.softplus(x[...,self.dim_x:])

        return collections.namedtuple("x", "x_mu x_std")._make([x_mu,x_std])

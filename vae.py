# Classes
import torch
import pyro
import pyro.distributions as dist

class VAE(torch.nn.Module):
    
    def __init__(self,params,encoder,decoder):
        super().__init__()
        
        # Parameters
        self.use_cuda = params['use_cuda']
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.dim_z  = params['DIM_Z']

        # Instantiate the encoder and decoder
        self.decoder = decoder
        self.encoder = encoder
        
        if(self.use_cuda):
            self.cuda()
        
    def guide(self, imgs=None):
        """ 1. run the inference 
            2. sample latent variables 
        """       
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            observed = False
            imgs = torch.zeros(8,self.ch,self.height,self.width)
            if(self.use_cuda):
                imgs=imgs.cuda()
        else:
            observed = True
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#

        pyro.module("encoder", self.encoder)
        
        batch_size,ch,width,height = imgs.shape
        
        with pyro.plate('batch_size', batch_size, dim=-1):
            z = self.encoder(imgs)
            pyro.sample('z_latent', dist.Normal(z.z_mu,z.z_std).to_event(1))
            
    def model(self, imgs=None):
        """ 1. sample the latent from the prior:
            2. runs the generative model
            3. score the generative model against actual data 
        """
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            observed = False
            imgs = torch.zeros(8,self.ch,self.height,self.width)
            if(self.use_cuda):
                imgs=imgs.cuda()
        else:
            observed = True
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#
        
        pyro.module("decoder", self.decoder)

        zero  = imgs.new_zeros(1)
        one   = imgs.new_ones(1)
        
        batch_size,ch,width,height = imgs.shape
        
        with pyro.plate('batch_size', batch_size, dim=-1):
            z = pyro.sample('z_latent', dist.Normal(zero,one).expand([self.dim_z]).to_event(1))
            x = self.decoder(z) #x_mu is between 0 and 1
            pyro.sample('obs', dist.Normal(x.x_mu,x.x_std).to_event(1), obs=imgs.view(batch_size,-1))
        return x
    
    def reconstruct(self,imgs):
        z = self.encoder(imgs)
        x = self.decoder(z.z_mu)
        return x.x_mu.view_as(imgs)

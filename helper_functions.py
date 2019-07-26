# Helper functions
import torch
import pickle
from torchvision import utils
import matplotlib.pyplot as plt
import pyro
import torch.nn.functional as F


    
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)
    
def show_batch(images,nrow=4,npadding=10,title=None):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    batch, ch, width, height = images.shape
    if(images.device != "cpu"):
        images=images.cpu()
    grid = utils.make_grid(images,nrow, npadding, normalize=True, range=None, scale_each=True, pad_value=1)       
    fig = plt.imshow(grid.detach().numpy().transpose((1, 2, 0))) 
    if(isinstance(title, str)):
        plt.title(title)
    return fig
       
    
def train(svi, loader, use_cuda=False,verbose=False):
    epoch_loss = 0.
    for x, _ in loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        
        loss = svi.step(x)
        if(verbose):
            print("loss=%.5f" %(loss))
        epoch_loss += loss

    # return epoch loss
    return epoch_loss / len(loader.dataset) 

def train_VAE_pytorch(vae, loader, optimizer, use_cuda=False, verbose=False):
    epoch_loss = 0.
    for x, _ in loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        
        optimizer.zero_grad()
        
        # run the model
        z = vae.encoder(x)
        z_tmp = z.z_mu + z.z_std * torch.randn_like(z.z_std)
        x_rec = vae.decoder(z_tmp)
        rec_loss   = 10*F.mse_loss(x_rec.x_mu.view(-1,784),x.view(-1,784), reduction='sum')
        # prior_loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        z_var = z.z_std.pow(2)
        KL = - 0.5 * torch.sum(1 + 2*torch.log(z.z_std)-z.z_mu.pow(2)-z.z_std.pow(2))
        #print("KL",KL)
        loss = rec_loss + KL
        
        if(verbose):
            print("loss=%.5f" %(loss.item()))
            
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    # return epoch loss
    return epoch_loss / len(loader.dataset) 



def evaluate(svi, loader, use_cuda=False, verbose=False):
    epoch_loss = 0.
    for x, _ in loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        loss = svi.evaluate_loss(x)
        if(verbose):
            print("loss=%.5f" %(loss))
        epoch_loss += loss
    return epoch_loss / len(loader.dataset) 




def save_obj(obj,root_dir,name):
    with open(root_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(root_dir,name):
    with open(root_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_model(model, root_dir, name):
    full_file_path= root_dir + name + '.pkl'
    torch.save(model.state_dict(),full_file_path)
    
def load_model(model, root_dir, name):
    full_file_path= root_dir + name + '.pkl'
    model.load_state_dict(torch.load(full_file_path))

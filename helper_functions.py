# Helper functions
import torch
import pickle
from torchvision import utils
import matplotlib.pyplot as plt
import pyro
from sklearn.manifold import TSNE

    
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)

def show_batch(images,nrow=4,npadding=10):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    batch, ch, width, height = images.shape
    if(images.device != "cpu"):
        images=images.cpu()
    grid = utils.make_grid(images,nrow, npadding, normalize=True, range=None, scale_each=True, pad_value=1)       
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0))) 
    

def show_2_batch(images1,images2,nrow=4,npadding=10):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert(images1.shape == images2.shape)
    if(images1.device != "cpu"):
        images1=images1.cpu()
    if(images1.device != "cpu"):
        images2=images2.cpu()
    tmp = torch.cat((images1,images2),dim=0)
    grid = utils.make_grid(tmp,nrow, npadding, normalize=True, range=None, scale_each=True, pad_value=1)       
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))     
    
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

# Helper functions
import torch
import pickle
from torchvision import utils
import matplotlib.pyplot as plt
import pyro
import torch.nn.functional as F
import numpy as np


    
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


def lr_finder(svi, loader=None, lr_start=1E-6, lr_end=None, lr_multiplier=1.5, N_max_epoch=None, use_cuda=False):
    """ Usage:
        hist_loss,hist_lr = lr_finder(svi, 
                                      loader=testloader, 
                                      lr_start=1E-6, 
                                      lr_multiplier=1.5, 
                                      N_max_epoch=30, 
                                      use_cuda=params['use_cuda'])
    """
    
    if(lr_end is None and N_max_epoch is None):
        raise Exception ("Either lr_end or N_max_epoch need to be scecified to avoid infinite loop")
        return None,None
    elif(not(lr_end is None)):
        N1 = int(np.log(np.float(lr_end)/lr_start)/np.log(np.float(lr_multiplier)))+1
        if(N_max_epoch is None):
            N_max_epoch = N1
        else:
            N_max_epoch = np.min(N_max_epoch,N1)
            
    # replace the optimizer with the scheduler if necessary
    if(isinstance(svi.optim,pyro.optim.optim.PyroOptim)):
        optimizer_args    = svi.optim.pt_optim_args    
        optimizer_pointer = svi.optim.pt_optim_constructor
        scheduler_args = {'optimizer': optimizer_pointer, 'step_size' : 1, 'gamma' : lr_multiplier, 'optim_args' : optimizer_args}
        svi.optim = pyro.optim.StepLR(scheduler_args)
    
    # check that now I have a pyro scheduler
    assert(isinstance(svi.optim,pyro.optim.lr_scheduler.PyroLRScheduler))
    
    hist_loss, hist_lr = [],[]
    for epoch in range(0,N_max_epoch): 
    
        # perform the training
        loss_curr = one_epoch_train(svi, loader, use_cuda=use_cuda, verbose=False)
        
        # get the current lr
        lr_curr = next(iter(svi.optim.optim_objs.values())).get_lr()[0] 
        
        # step the scheduler (this assumes that scheduler steps one for epoch)
        svi.optim.step() 
                    
        print("[epoch %03d] train loss: %.4f lr: %.4e" % (epoch, loss_curr, lr_curr))
    
        hist_loss.append(loss_curr)  
        hist_lr.append(lr_curr)

        if(np.isnan(loss_curr)):
            return hist_loss,hist_lr
    
    return hist_loss,hist_lr

    
def one_epoch_train(svi, loader, use_cuda=False,verbose=False):
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

def one_epoch_train_VAE_pytorch(vae, loader, optimizer, use_cuda=False, verbose=False):
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



def one_epoch_evaluate(svi, loader, use_cuda=False, verbose=False):
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


### HOW TO DO A LR_FINDER
#### preparation
###pr_hist_loss = []
###pr_hist_lr = []
###for epoch in range(0,30):
###    
###    if(isinstance(svi.optim, pyro.optim.lr_scheduler.PyroLRScheduler)):
###        svi.optim.step(epoch=epoch) # step the PYRO LR_scheduler 
###
###    loss_curr = train(svi, trainloader, use_cuda=params['use_cuda'], verbose=False)
###    
###    if(isinstance(svi.optim, pyro.optim.lr_scheduler.PyroLRScheduler)):
###        pt_scheduler = next(iter(svi.optim.optim_objs.values()))
###        lr_curr = pt_scheduler.get_lr()[0] 
###    elif(isinstance(svi.optim, pyro.optim.optim.PyroOptim)):
###        lr_curr = svi.optim.pt_optim_args['lr']
###                    
###    print("[epoch %03d] train loss: %.4f lr: %.4e" % (epoch, loss_curr, lr_curr))
###    
###    pr_hist_loss.append(loss_curr)  
###    pr_hist_lr.append(lr_curr)
###
###    if(np.isnan(loss_curr)):
###        break 
###        
###    imgs_rec = vae.reconstruct(imgs_test)
###    show_batch(imgs_rec,nrow=4,npadding=4,title="epoch = "+str(epoch))
###    plt.savefig("PYRO_rec_epoch_"+str(epoch)+".png")
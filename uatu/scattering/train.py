import torch
import torch.nn.functional as F
from .attack import compute_attacked_map
import sys

def train(model, device, train_loader, optimizer, epoch, scattering, print_every = 1000, loss = 'mae'):
    model.train()

    loss_fn = F.l1_loss if loss == 'mae' else F.mse_loss
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.squeeze(data).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx*len(data) / len(train_loader), loss.item()))
            sys.stdout.flush()

# TODO i could decorate this like I did in the tf stuff
def adv_train(model, device, train_loader, optimizer, epoch, scattering, print_every = 1000, loss = 'mae'):
    model.train()

    loss_fn = F.l1_loss if loss == 'mae' else F.mse_loss
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.squeeze(data).to(device), target.to(device)
        #output = model(scattering(data))
        #orig_loss = loss_fn(output, target)

        #with torch.no_grad():
        adv_data = compute_attacked_map(model,scattering, loss_fn, data, target) 
        
        #with torch.no_grad(): # don't wanna run gradients back through the perturbation
        #    model.eval()
        adv_output = model(scattering(adv_data))
        #    model.train()
            
        adv_loss = loss_fn(adv_output, target)

        #loss = 0.5*orig_loss + 0.5*adv_loss
        loss = adv_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()
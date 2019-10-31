import torch
import torch.nn.functional as F
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
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()

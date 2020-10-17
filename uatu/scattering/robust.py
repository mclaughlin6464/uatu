### Similar to attacks, make robust clones of trainign examples
import torch
from .attack import fgsm_attack, log_barrier

# TODO would be nice to get a depth of embedding
def get_embedding(x, model, scattering = lambda x: x):
    #x = scattering(x)
    x = x.view(-1, model.K, model.input_size, model.input_size)#

    x = model.init_conv(x)

    for i,l in enumerate(model.layers):
        x = l(x)

    return model.avgpool(x)

def get_gupta_embedding(x, model, scattering = lambda x:x):
    x = x.view(-1, model.K, model.input_size, model.input_size)#

    for i, l in enumerate(model.layers):
        x = l(x)
                        
    # confused if this is necessary?
    x = x.transpose(1,3).contiguous()
    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = model.fc1(x)
    x = model.relu(x)#print(x.shape)
    x = model.fc2(x)
    x = model.relu(x)
    #x = model.fc3(x)
    #print(x.shape)
        
    return x

def get_fluri_embedding(x, model, scattering = lambda x:x):
    x = x.view(-1, model.K, model.input_size, model.input_size)#

    for i, l in enumerate(model.layers):
        x = l(x)
                        
    # confused if this is necessary?
    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = model.fc1(x)
    #x = model.relu(x)#print(x.shape)
    #x = model.fc2(x)
    #print(x.shape)
        
    return x


def compute_robust_map(scattering, device, model, x0, xt, learning_rate= 5e-3, lr_decay = 0.2, n_steps = 200, update_steps = 50,\
                         get_embedding=get_embedding): #use_log_barrier = True, log_eps = 1.5)

    # Send the data and label to the device
    x0, xt = x0.to(device), xt.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_x0 = x0.clone()
    perturbed_x0.requires_grad = True
    # Forward pass the data through the model
    #get_embedding = get_gupta_embedding
    init_pred = get_embedding(xt, model, scattering)

    for i in range(n_steps):
        optimizer = torch.optim.Adam([perturbed_x0], lr=learning_rate)#, weight_decay=1e-9)

        optimizer.zero_grad()

        output = get_embedding(perturbed_x0, model, scattering)
        loss = (output-init_pred).norm() + log_barrier(perturbed_x0, x0, eps=0.5, lam = 1e2) 
        loss.backward(retain_graph = True)

        optimizer.step()

        if i%update_steps == 0 and i>0:
            learning_rate*=lr_decay 
        
    return perturbed_x0#, init_pred, output

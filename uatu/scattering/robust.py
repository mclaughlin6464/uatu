### Similar to attacks, make robust clones of trainign examples
import torch
from .attack import fgsm_attack, log_barrier


# TODO would be nice to get a depth of embedding
def get_embedding(x, model, scattering = lambda x: x):
    x = scattering(x)

    x = x.view(-1, model.K, model.input_size, model.input_size)#

    x = model.init_conv(x)

    for i,l in enumerate(model.layers):
        x = l(x)

    return model.avgpool(x)

def compute_robust_map(scattering, device, model, x0, xt, learning_rate= 1e-3, lr_decay = 0.9, n_steps = 200, update_steps = 50): #use_log_barrier = True, log_eps = 1.5)

    # Send the data and label to the device
    x0, xt = x0.to(device), xt.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_x0 = x0.clone()

    perturbed_x0.requires_grad = True
    scattering.requires_grad = False
    # Forward pass the data through the model
    init_pred = get_embedding(xt, model, scattering)

    for i in range(n_steps):
        output = get_embedding(perturbed_x0, model, scattering)
        # TODO put power spectrum here too? 
        loss = (output-init_pred).norm() + log_barrier(perturbed_x0, x0)
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward(retain_graph = i< n_steps-1)

        # Collect datagrad
        # TODO can I use an optimizer here, like Adam, etc?
        x0_grad = perturbed_x0.grad.data
        # Call FGSM Attack
        perturbed_x0 = fgsm_attack(perturbed_x0, learning_rate, x0_grad)
        perturbed_x0 = torch.autograd.Variable(perturbed_x0.data, requires_grad=True)

        if i>0 and i%update_steps == 0:
            learning_rate*=lr_decay
        
    return perturbed_x0#, init_pred, output

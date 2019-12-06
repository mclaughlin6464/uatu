import torch
import torch.nn as nn
import torch.functional as F

def log_barrier(x_p, x_o, eps=2.5, lam=1e6):
    norm = (x_p - x_o).norm(p=float('Inf')) + 1e-6
    return -torch.log(eps - norm )/lam

def fgsm_attack(image, eps, data_grad):

    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image.detach() - eps*sign_data_grad.detach()

    # Adding clipping to maintain [0,1] range
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def compute_attacked_map(model, scattering, cost_fn, data, target, use_log_barrier = False, n_steps = 1):#, log_eps = 1.5)

    # Send the data and label to the device
    #data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True
    scattering.requires_grad = False
    # Forward pass the data through the model
    init_pred = model(scattering(perturbed_data))
    # TODO put this in a loop with the log barrier
    # Calculate the loss
    output = init_pred
    for i in range(n_steps):
        loss = cost_fn(output, target)

        if use_log_barrier:
            loss+=log_barrier(data, perturbed_data)

        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward(retain_graph = i<n_steps-1)

# Collect datagrad
        data_grad = perturbed_data.grad.data

        # Call FGSM Attack
        epsilon = 1e-3 
# sign change is important to make it a gradient ascent
        perturbed_data = fgsm_attack(perturbed_data, epsilon, -1*data_grad)

        # Re-classify the perturbed image
        if i < n_steps-1:
            output = model(scattering(perturbed_data))

    return perturbed_data#, init_pred, output

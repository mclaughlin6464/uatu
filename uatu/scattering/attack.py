import torch
import torch.nn as nn
import torch.nn.functional as F

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

def compute_attacked_map(model, scattering, cost_fn, data, target, use_log_barrier = False, n_steps = 5):#, log_eps = 1.5)
    perturbed_data = data.clone()
    for i in range(n_steps):

        perturbed_data.requires_grad_()
        
        with torch.enable_grad():
            output = model(scattering(perturbed_data))

            loss = cost_fn(output, target)

            if use_log_barrier:
                loss+=log_barrier(data, perturbed_data)

        data_grad = torch.autograd.grad(loss, [perturbed_data])[0]
                    
        epsilon = 2e-4
        # sign change is important to make it a gradient ascent
        perturbed_data = fgsm_attack(perturbed_data, epsilon, -1*data_grad)

    return perturbed_data#, init_pred, output

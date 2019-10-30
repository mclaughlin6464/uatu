import torch
#import torch.Functional as F
import numpy as np

def val_test(model, device, test_loader, scattering):

    model.eval()
    perc_error = []
    rmse = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data)).item()
            target = target.item()
            perc_error.append(np.array(np.abs(output- target) / (target)))
            rmse.append(np.array(output - target))

    acc = np.abs(np.vstack(perc_error).mean(axis=0))
    print('Om: %.2f%%, s8: %.2f%% accuracy' % (100 * acc[0], 100 * acc[1]) )
    rmse = np.sqrt(np.mean(np.vstack(rmse) ** 2, axis=0))
    print('RMSE: %.4f, %.4f' % (rmse[0], rmse[1]) )
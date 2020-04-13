import torch
#import torch.Functional as F
import numpy as np
import h5py
from ..watchers.test import key_func

def val_test(model, device, test_loader, scattering=lambda x:x):

    model.eval()
    perc_error = []
    rmse = []

    with torch.no_grad():
        for data, target in test_loader:
            if len(data.shape) > 3:
                data = torch.squeeze(data, 3)
            data = data.to(device)#, target.to(device)
            output = model(scattering(data)).cpu()
            perc_error.append(np.array(np.abs(output- target) / (target)))
            rmse.append(np.array(output - target))

    acc = np.abs(np.vstack(perc_error).mean(axis=0))
    print('Om: %.2f%%, s8: %.2f%% accuracy' % (100 * acc[0], 100 * acc[1]) )
    rmse = np.sqrt(np.mean(np.vstack(rmse) ** 2, axis=0))
    print('RMSE: %.4f, %.4f' % (rmse[0], rmse[1]) )

def test(model, device, test_loader, output_fname, scattering=lambda x:x):

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            if len(data.shape)>3:
                data = torch.squeeze(data,3)

            data = data.to(device)
            output = model(scattering(data)).cpu().numpy()
            target= target.numpy()
            for y_pred, y_true in zip(output, target):
                key = key_func(y_true.reshape((1,-1)))
                with h5py.File(output_fname) as f:
                    if key in f.keys():
                        grp = f[key]
                    else:
                        grp = f.create_group(key)
                    n_prev = len(grp.keys())

                    grp.create_dataset('Map_%04d'%(n_prev), data = y_pred)

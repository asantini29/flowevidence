from flowevidence.encode import SharedModelAE, SharedModelDecoder, SharedModelEncoder, MaskedAutoEncoder
from flowevidence.utils import *
import torch
import matplotlib.pyplot as plt
import corner
from eryn.backends import HDFBackend

import argparse

use_gpu = True

def clean_chain(chain):
        ndim = chain.shape[1]
        naninds = np.logical_not(np.isnan(chain[:, 0].flatten()))
        samples_out = np.zeros((chain[:,0].flatten()[naninds].shape[0], ndim))  # init the chains to plot\n",
        for d in range(ndim):
            givenparam = chain[:, d].flatten()
            samples_out[:, d] = givenparam[
                np.logical_not(np.isnan(givenparam))
            ]
        return samples_out

parser = argparse.ArgumentParser(description="test the autoencoder on a mcmc backend")

parser.add_argument("-dev", "--dev", help="Cuda Device", required=False, type=int, default=0)
parser.add_argument("-train", "--train", help="train the encoder", required=False, action='store_true')

args = vars(parser.parse_args())
gpu_index = args['dev']

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() and use_gpu else 'cpu')
train = args['train']

if __name__ == '__main__':
    backend = HDFBackend("backend.h5")

    samples_all = backend.get_chain()['gauss']
    ns, nt, nw, nl, nd = samples_all.shape
    samples = samples_all[:, 0].reshape(-1, nl*nd)

    max_dim = samples.shape[1]
    latent_dim = nd # avoid compression loss
    hidden_dim = 256
    dropout = 0.1
    split_ratio = 0.7
    dtype = torch.float64
    verbose = True  

    autoencoder = MaskedAutoEncoder(max_model_dim=max_dim, 
                                    latent_dim=latent_dim,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout, 
                                    device=device, 
                                    dtype=dtype, 
                                    verbose=verbose
                                    )

    posterior_samples = torch.tensor(samples, dtype=dtype)
    latent_samples, q1, q2 = normalize_minmax(posterior_samples)
    latent_target = latent_samples

    shuffled_samples = shuffle(latent_samples)
    if split_ratio:
        train_samples, val_samples = split(shuffled_samples, split_ratio)
    else:
        train_samples, val_samples = shuffled_samples, None
    
    batch_size = val_samples.shape[0] // 20 if train_samples is not None else 128
    batch_size = 512
    train_loader, val_loader = create_data_loaders(train_samples, val_samples, batch_size=batch_size, num_workers=0)

    lambda_L1 = 0.0
    if train: 
        autoencoder.train(train_loader, 
                        val_loader, 
                        epochs=5000, 
                        lr=1e-3, 
                        lambda_L1=lambda_L1,
                        early_stopping=True
                        )
    else:
        autoencoder.load_model(path="autoencoder.pth")

    encoded = autoencoder.encode(latent_target.to(device))
    decoded = autoencoder.decode(encoded)
   
    print('nans predicted by the autoencoder: %.i' % torch.isnan(decoded).sum().item())
    print('nans present in the target: %.i' % torch.isnan(latent_target).sum().item())
   
    target_clean = denormalize_minmax(latent_target, q1, q2)
    target_clean = target_clean.cpu().detach().numpy()
    target_clean = target_clean.reshape(-1, nd)
    target_clean = clean_chain(target_clean)
    
    fig = corner.corner(target_clean, color="k", density=True)
    decoded_clean = denormalize_minmax(decoded, q1, q2)
    decoded_clean = decoded_clean.cpu().detach().numpy()
    decoded_clean = decoded_clean.reshape(-1, nd)
    decoded_clean = clean_chain(decoded_clean)
    
    fig = corner.corner(decoded_clean, color='red', fig=fig, density=True)
    plt.savefig('diagnostic/corner_decoded.pdf')
    plt.close()
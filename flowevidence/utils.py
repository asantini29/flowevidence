import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd.profiler as profiler
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform, ReversePermutation, AffineCouplingTransform, RandomPermutation
from nflows.nn.nets import MLP, ResidualNet
from torch.amp import GradScaler, autocast
import gc
import warnings
import matplotlib.pyplot as plt
from corner import corner
import logging

def setup_logging(verbose=False):
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

# Define preprocessing functions
def standardize(samples):
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    normalized_samples = (samples - mean) / std
    return normalized_samples, mean, std

def normalize(samples):
    range = samples.max(dim=0).values - samples.min(dim=0).values
    minimum = samples.min(dim=0).values

    normalized_samples = (samples - minimum) / range
    return normalized_samples, minimum, range

def destandardize(samples, mean, std):
    return samples * std + mean

def denormalize(samples, minimum, range):
    return samples * range + minimum

def shuffle(samples):
    indices = torch.randperm(samples.size(0))
    return samples[indices]

def split(samples, train_ratio=0.8):
    num_train = int(train_ratio * samples.size(0))
    train_samples = samples[:num_train]
    val_samples = samples[num_train:]
    return train_samples, val_samples

def create_data_loaders(train_samples, val_samples, batch_size=256, num_workers=4):
    train_dataset = TensorDataset(train_samples)
    val_dataset = TensorDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# define the flow model
def get_flow(num_dims, num_flow_steps, use_nvp=False, device='cpu'):
    # Define the base distribution
    base_distribution = StandardNormal(shape=[num_dims])

    # Define the transforms
    transforms = []
    for _ in range(num_flow_steps):
        #transforms.append(ReversePermutation(features=num_dims))
        transforms.append(RandomPermutation(features=num_dims))

        if use_nvp:
            logging.warning("NVP seemed not to converge as well as MAF, use it mostly for testing at the moment")
            transforms.append(AffineCouplingTransform(
                mask=torch.arange(0, num_dims) % 2,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features,
                    out_features,
                    hidden_features=128,
                    num_blocks=2,
                    activation=nn.ReLU()
                )
            ))     
            
        else:
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=num_dims, 
                    hidden_features=128, 
                    dropout_probability=0.3,
                    use_batch_norm=False,
                    )
                )
   

    # Combine the transforms into a composite transform
    transform = CompositeTransform(transforms)

    # Create the flow model
    flow = Flow(transform, base_distribution).to(device)

    return flow

def l2_regularization(model, lambdaL2):
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    return lambdaL2 * l2_reg

# Define a loss function (negative log likelihood)
def loss_fn(model, x):
    return -model.log_prob(x).mean()

def cornerplot_training(samples, target_distribution=None, epoch=0, plot_dir='./', savename='corner'):
    if target_distribution is not None:
        fig = corner(target_distribution, bins=50, color='k')
        fig = corner(samples, bins=50, color='r', fig=fig)

        handles = [
        plt.Line2D([], [], color='k', label='Target Distribution'),
        plt.Line2D([], [], color='r', label='Flow @ epoch ' + str(epoch))
    ]
    else:
        fig = corner(samples, bins=50, color='r')
        handles = [
        plt.Line2D([], [], color='r', label='Flow @ epoch ' + str(epoch))
    ]

    axes = fig.axes  # Get the axes of the figure
    axes[-1].legend(handles=handles, loc="upper right")  # Add legend to the last axis
    plt.savefig(plot_dir + savename)
    plt.close(fig)

def lossplot(epochs_losses, train_losses, val_losses, plot_dir='./', savename='losses'):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(epochs_losses, label='total')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_dir + savename)
    plt.close(fig)
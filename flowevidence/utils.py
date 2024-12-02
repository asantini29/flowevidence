# coding: utf-8

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation, ReversePermutation
from nflows.nn.nets import ResidualNet

import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import logging
from .trasforms import get_transform

def setup_logging(verbose=False):
        """
        Configures the logging settings for the application.
        
        Args:
            verbose (bool): If True, sets the logging level to INFO. Otherwise, sets it to WARNING.
        """
        
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )


# Define preprocessing functions
def normalize_gaussian(samples: torch.tensor) -> tuple[torch.tensor: torch.tensor, torch.tensor]:
    """
    Standardizes the given samples by removing the mean and scaling to unit variance. Add masking operations to deal with NaN values, for example introduced by RJMCMC.
    
    Args:
        samples (torch.Tensor): A tensor containing the samples to be standardized.
    
    Returns:
        normalized_samples (torch.Tensor): The standardized samples.
        mean (torch.Tensor): The mean of the original samples.
        std (torch.Tensor): The standard deviation of the original samples.
    """
    finite_mask = torch.isfinite(samples)  # Boolean mask for finite values

    # Replace non-finite values with 0 (or any placeholder value that won't affect sums)
    finite_samples = samples.clone()
    finite_samples[~finite_mask] = 0.0

    # Count of finite values per column
    count_finite = finite_mask.sum(dim=0)

    # Compute sum of finite values per column
    sum_finite = finite_samples.sum(dim=0)

    # Column-wise mean: Avoid division by zero
    mean = sum_finite / count_finite
    mean[count_finite == 0] = float('nan')  # Set mean to NaN where no finite values exist

    # Compute variance and standard deviation
    squared_diff = (samples - mean.unsqueeze(0)) ** 2
    squared_diff[~finite_mask] = 0.0  # Ignore non-finite values
    variance = squared_diff.sum(dim=0) / count_finite
    variance[count_finite == 0] = float('nan')  # Set variance to NaN where no finite values exist
    std = variance.sqrt()  # Standard deviation

    normalized_samples = (samples - mean) / std

    return normalized_samples, mean, std

def normalize_minmax(samples: torch.tensor) -> tuple[torch.tensor: torch.tensor, torch.tensor]:
    """
    Normalizes the given samples by scaling to the range [0, 1].

    Args:
        samples (torch.Tensor): A tensor containing the samples to be normalized.
    
    Returns:
        normalized_samples (torch.Tensor): The normalized samples.
        minimum (torch.Tensor): The minimum value of the original samples.
        range (torch.Tensor): The range of the original samples.
    """
    finite_mask = torch.isfinite(samples)  # Boolean mask for finite values

    # Replace non-finite values with 0 to avoid affecting min and max calculations
    finite_samples = samples.clone()
    finite_samples[~finite_mask] = 0.0

    # Compute per-column min and max while ignoring non-finite values
    min_values = torch.where(finite_mask, samples, float('inf')).min(dim=0).values
    max_values = torch.where(finite_mask, samples, float('-inf')).max(dim=0).values

    # Compute range, ensuring no division by zero
    range_values = max_values - min_values
    range_values[range_values == 0] = float('nan')  # Handle zero range gracefully

    # Normalize samples
    normalized_samples = (samples - min_values) / range_values

    # Return normalized samples, along with the min and range used for normalization
    return normalized_samples, min_values, range_values

def denormalize_gaussian(samples: torch.tensor, 
                mean: torch.tensor,
                std: torch.tensor
                ) -> torch.tensor:
    """
    Denormalizes the given samples by adding the mean and scaling by the standard deviation.

    Args:
        samples (torch.Tensor): A tensor containing the samples to be destandardized.
        mean (torch.Tensor): The mean of the original samples.
        std (torch.Tensor): The standard deviation of the original samples.
    
    Returns:
        destandardized_samples (torch.Tensor): The destandardized samples.
    """

    return samples * std + mean

def denormalize_minmax(samples: torch.tensor, 
                minimum: torch.tensor,
                range: torch.tensor
                ) -> torch.tensor:
    """
    Denormalizes the given samples by scaling by the range and adding the minimum.

    Args:
        samples (torch.Tensor): A tensor containing the samples to be denormalized.
        minimum (torch.Tensor): The minimum value of the original samples.
        range (torch.Tensor): The range of the original samples.
    
    Returns:
        denormalized_samples (torch.Tensor): The denormalized samples.
    """
    return samples * range + minimum

def shuffle(samples: torch.tensor) -> torch.tensor:
    """
    Shuffles the given tensor of samples along the first dimension.
    
    Args:
        samples (torch.Tensor): A tensor containing the samples to be shuffled.
    
    Returns:
        torch.Tensor: A tensor with the samples shuffled along the first dimension.
    """

    indices = torch.randperm(samples.size(0))
    return samples[indices]

def split(samples: torch.tensor, 
          train_ratio: float = 0.8
          ) -> tuple[torch.tensor, torch.tensor]:
    """
    Splits the given samples into training and validation sets based on the specified training ratio.
    
    Args:
        samples (Tensor): The input samples to be split.
        train_ratio (float, optional): The ratio of samples to be used for training. Defaults to 0.8.
    
    Returns:
        train_samples (Tensor): The training samples.
        val_samples (Tensor): The validation samples.
    """

    num_train = int(train_ratio * samples.size(0))
    train_samples = samples[:num_train]
    val_samples = samples[num_train:]
    return train_samples, val_samples

def create_data_loaders(train_samples: torch.tensor, 
                        val_samples: torch.tensor,
                        batch_size: int = 256, 
                        num_workers: int = 0,
                        pin_memory: bool = True
                        ) -> tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for training and validation datasets.
    
    Args:
        train_samples (Tensor): The training samples.
        val_samples (Tensor): The validation samples.
        batch_size (int, optional): Number of samples per batch to load. Default is 256.
        num_workers (int, optional): How many subprocesses to use for data loading. Default is 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory. Default is True.
    
    Returns:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
    """

    train_dataset = TensorDataset(train_samples)
    val_dataset = TensorDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader

# define the flow model
def get_flow(num_dims: int, 
             num_flow_steps: int, 
             transform_type: str = 'maf', 
             transform_kwargs: dict = {},
             device: str | torch.device = 'cpu'
             ) -> Flow:
    # Define the base distribution
    base_distribution = StandardNormal(shape=[num_dims])

    # Define the transforms
    transforms = []
    transform_class, default_kwargs = get_transform(transform_type)
    transform_kwargs = {**default_kwargs, **transform_kwargs}

    if transform_type == 'nvp':
        logging.warning("NVP is still under development and may not work as expected.") 
    else:
        transform_kwargs['features'] = num_dims

    mask = torch.arange(num_dims) % 2

    if 'mask' in transform_kwargs and transform_kwargs['mask'] is None:
        transform_kwargs['mask'] = mask 
    
    for _ in range(num_flow_steps):
        #transforms.append(ReversePermutation(features=num_dims))
        transforms.append(RandomPermutation(features=num_dims))
        transforms.append(transform_class(**transform_kwargs))

        if transform_type == 'rqs':
            transforms.append(ReversePermutation(features=num_dims))
            mask = 1 - mask
   
    # Combine the transforms into a composite transform
    transform = CompositeTransform(transforms)

    # Create the flow model
    device = torch.device(device) if isinstance(device, str) else device
    flow = Flow(transform, base_distribution).to(device)

    return flow

def l2_regularization(model: nn.Module, 
                      lambdaL2: float
                      ) -> torch.Tensor:
    """
    Add L2 regularization to the model.

    Args:
        model (nn.Module): The model to which L2 regularization will be added.
        lambdaL2 (float): The regularization strength.
    
    Returns:
        torch.Tensor: The L2 regularization
    """
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    return lambdaL2 * l2_reg

# Define a loss function (negative log likelihood)
def loss_fn(model: nn.Module, 
            x: torch.Tensor
            ) -> torch.Tensor:
    """
    Computes the negative log likelihood of the given samples under the model.
    
    Args:
        model (nn.Module): The model to evaluate.
        x (torch.Tensor): The samples to evaluate.
    
    Returns:
        torch.Tensor: The negative log likelihood of the samples under the model.
    """

    return -model.log_prob(x).mean()

def cornerplot_training(samples: np.ndarray,    
                        target_distribution: np.ndarray = None,
                        epoch: int = 0,
                        plot_dir: str = './',
                        savename: str = 'corner'
                        ):
    """
    Generates a corner plot for the given samples and optionally overlays it with a target distribution.
    
    Args:
        samples (np.ndarray): The samples to be plotted.
        target_distribution (np.ndarray, optional): The target distribution to overlay on the plot. Defaults to None.
        epoch (int, optional): The current epoch number, used for labeling. Defaults to 0.
        plot_dir (str, optional): The directory where the plot will be saved. Defaults to './'.
        savename (str, optional): The name of the saved plot file. Defaults to 'corner'.
    """
    color_target = 'k'
    color_samples = "#5790fc"
    if target_distribution is not None:
        fig = corner(target_distribution, bins=50, color=color_target)
        fig = corner(samples, bins=50, color=color_samples, fig=fig)

        handles = [
        plt.Line2D([], [], color=color_target, label='Target \n Distribution'),
        plt.Line2D([], [], color=color_samples, label='Training @ \n epoch ' + str(epoch))
    ]
    else:
        fig = corner(samples, bins=50, color=color_samples)
        handles = [
        plt.Line2D([], [], color=color_samples, label='Flow @ \n epoch ' + str(epoch))
    ]
    
    ndims = samples.shape[1] # Number of dimensions in the samples
    axes = np.array(fig.axes).reshape(ndims, ndims)  # Get the axes of the figure
    axes[0, 1].legend(handles=handles, loc="upper left")  # Add legend to the last axis
    #plt.tight_layout()
    plt.savefig(plot_dir + savename)
    plt.close(fig)

def lossplot(epochs: np.ndarray | list, 
             train_losses: np.ndarray | list,
             val_losses: np.ndarray | list,
             plot_dir: str = './',
             savename: str = 'losses'
             ):
    """
    Plots the training and validation losses over epochs and saves the plot as an image file.
    
    Args:
        epochs (list or array-like): List or array of epoch numbers.
        train_losses (list or array-like): List or array of training losses for each epoch.
        val_losses (list or array-like): List or array of validation losses for each epoch.
        plot_dir (str, optional): Directory where the plot image will be saved. Default is './'.
        savename (str, optional): Name of the saved plot image file. Default is 'losses'.
    """
    #ensure they are arrays
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    fig = plt.figure(figsize=(12, 8))

    # set an offset to make all the values positive and allow the semilogy plot
    # offset = np.abs(min(np.min(train_losses), np.min(val_losses))) + 1
    # train_losses += offset
    # val_losses += offset

    plt.plot(epochs, train_losses, '-x', label='Training')
    plt.plot(epochs, val_losses, '-x', label='Validation')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(plot_dir + savename)
    plt.close(fig)

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

class EarlyStopping:
    """
    Early stopping class to stop training the flow model when the validation loss does not improve.
    
    Args:
        patience (int): Number of epochs to wait before stopping training. Default is 50.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 1e-6.
    
    Methods:
        __call__(val_loss):
            Checks if the validation loss has improved.
    """

    def __init__(self, patience: int = 50, delta: float = 1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float):
        """
        Checks if the validation loss has improved.
        
        Args:
            val_loss (float): The validation loss to check.
        
        Returns:
            stop (bool): True if the validation loss has not improved for the specified number of epochs, False otherwise.
        """

        if np.abs(val_loss - self.best_loss) < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

    
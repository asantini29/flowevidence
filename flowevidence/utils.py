import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform, AffineCouplingTransform, RandomPermutation
from nflows.nn.nets import ResidualNet
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import logging

def setup_logging(verbose=False):
        """
        Configures the logging settings for the application.
        
        Args:
            verbose (bool): If True, sets the logging level to DEBUG. Otherwise, sets it to INFO.
        """
        
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

# Define preprocessing functions
def standardize(samples: torch.tensor) -> tuple[torch.tensor: torch.tensor, torch.tensor]:
    """
    Standardizes the given samples by removing the mean and scaling to unit variance.
    
    Args:
        samples (torch.Tensor): A tensor containing the samples to be standardized.
    
    Returns:
        normalized_samples (torch.Tensor): The standardized samples.
        mean (torch.Tensor): The mean of the original samples.
        std (torch.Tensor): The standard deviation of the original samples.
    """

    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    normalized_samples = (samples - mean) / std
    return normalized_samples, mean, std

def normalize(samples: torch.tensor) -> tuple[torch.tensor: torch.tensor, torch.tensor]:
    """
    Normalizes the given samples by scaling to the range [0, 1].

    Args:
        samples (torch.Tensor): A tensor containing the samples to be normalized.
    
    Returns:
        normalized_samples (torch.Tensor): The normalized samples.
        minimum (torch.Tensor): The minimum value of the original samples.
        range (torch.Tensor): The range of the original samples.
    """

    range = samples.max(dim=0).values - samples.min(dim=0).values
    minimum = samples.min(dim=0).values

    normalized_samples = (samples - minimum) / range
    return normalized_samples, minimum, range

def destandardize(samples: torch.tensor, 
                mean: torch.tensor,
                std: torch.tensor
                ) -> torch.tensor:
    """
    Destandardizes the given samples by adding the mean and scaling by the standard deviation.

    Args:
        samples (torch.Tensor): A tensor containing the samples to be destandardized.
        mean (torch.Tensor): The mean of the original samples.
        std (torch.Tensor): The standard deviation of the original samples.
    
    Returns:
        destandardized_samples (torch.Tensor): The destandardized samples.
    """

    return samples * std + mean

def denormalize(samples: torch.tensor, 
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
                        num_workers: int = 4
                        ) -> tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for training and validation datasets.
    
    Args:
        train_samples (Tensor): The training samples.
        val_samples (Tensor): The validation samples.
        batch_size (int, optional): Number of samples per batch to load. Default is 256.
        num_workers (int, optional): How many subprocesses to use for data loading. Default is 4.
    
    Returns:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
    """

    train_dataset = TensorDataset(train_samples)
    val_dataset = TensorDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# define the flow model
def get_flow(num_dims: int, 
             num_flow_steps: int, 
             use_nvp: bool = False, 
             device: str | torch.device = 'cpu'
             ) -> Flow:
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

def lossplot(epochs_losses: np.ndarray | list, 
             train_losses: np.ndarray |  list,
             val_losses: np.ndarray | list,
             plot_dir: str = './',
             savename: str = 'losses'
             ):
    """
    Plots the training and validation losses over epochs and saves the plot as an image file.
    
    Args:
        epochs_losses (list or array-like): List or array of total losses for each epoch.
        train_losses (list or array-like): List or array of training losses for each epoch.
        val_losses (list or array-like): List or array of validation losses for each epoch.
        plot_dir (str, optional): Directory where the plot image will be saved. Default is './'.
        savename (str, optional): Name of the saved plot image file. Default is 'losses'.
    """

    fig = plt.figure(figsize=(12, 8))
    plt.plot(epochs_losses, label='total')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_dir + savename)
    plt.close(fig)
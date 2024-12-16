import torch
import torch.nn as nn

from flowevidence.core import FlowContainer
from flowevidence.utils import split, create_data_loaders, normalize_minmax, denormalize_minmax
from sklearn.datasets import make_moons

import numpy as np
import matplotlib.pyplot as plt

import pysco
pysco.plot.default_plotting()
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU, 
                 dropout_prob=0.1, batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        current_dim = input_dim
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h),
                activation(),
            ])
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = h
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)

def rational_quadratic_spline(inputs, widths, heights, derivatives, inverse=False, 
                            min_bin_width=1e-3, min_bin_height=1e-3, 
                            min_derivative=1e-3):
    num_bins = widths.shape[-1]
    
    # Ensure constraints are satisfied
    widths = F.softmax(widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    derivatives = F.softplus(derivatives)
    derivatives = min_derivative + derivatives
    
    # Compute bin locations
    cumsum_widths = torch.cumsum(widths, dim=-1)
    cumsum_heights = torch.cumsum(heights, dim=-1)
    
    # Pad for convenience
    cumsum_widths = F.pad(cumsum_widths, (1, 0), value=0.0)
    cumsum_heights = F.pad(cumsum_heights, (1, 0), value=0.0)
    derivatives = F.pad(derivatives, (0, 1), value=min_derivative)
    
    # Make sure inputs have the right shape for comparison
    inputs = inputs.unsqueeze(-1)
    
    # Find which bin the inputs fall into
    if inverse:
        bin_idx = torch.sum(inputs > cumsum_heights[..., :-1], dim=-1)
    else:
        bin_idx = torch.sum(inputs > cumsum_widths[..., :-1], dim=-1)
    bin_idx = bin_idx.clamp(0, num_bins - 1)
    
    # Get bin parameters
    input_left = torch.gather(cumsum_widths, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_right = torch.gather(cumsum_widths, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    output_left = torch.gather(cumsum_heights, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    output_right = torch.gather(cumsum_heights, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    
    bin_width = input_right - input_left
    bin_height = output_right - output_left
    
    deriv_left = torch.gather(derivatives[..., :-1], -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    deriv_right = torch.gather(derivatives[..., 1:], -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    # Remove the extra dimension we added to inputs
    inputs = inputs.squeeze(-1)
    
    # Compute spline parameters
    t = (inputs - input_left) / bin_width if not inverse else (inputs - output_left) / bin_height
    t = t.clamp(0, 1)
    
    # Compute output and log_det using rational quadratic spline formula
    numerator = bin_height * (deriv_left * t.pow(2) + 2 * t * (1 - t))
    denominator = deriv_left + ((deriv_right - deriv_left) * t)
    
    outputs = output_left + numerator / denominator
    
    # Compute log determinant of Jacobian
    log_det = torch.log(bin_height) + 2 * torch.log(2 * t * (1 - t) * deriv_right + deriv_left)
    log_det = log_det - torch.log(denominator)
    
    return outputs, log_det.sum(dim=-1)

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_bins=8, mask_type='alternate',
                 activation=nn.ReLU, condition_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_bins = num_bins
        
        # Create alternating mask
        if mask_type == 'alternate':
            self.mask = torch.arange(input_dim) % 2
        elif mask_type == 'half':
            self.mask = torch.cat([torch.zeros(input_dim//2), torch.ones(input_dim - input_dim//2)])
        self.mask = self.mask.bool()
        
        # Determine dimensions for transform net
        masked_dim = self.mask.sum().item()
        conditioner_dim = (input_dim - masked_dim) + (condition_dim or 0)
        
        # Networks for predicting spline parameters
        transform_output_dim = masked_dim * num_bins * 3  # widths, heights, derivatives
        self.transform_net = ConditionalMLP(
            conditioner_dim, transform_output_dim, hidden_dims, activation
        )
        
    def forward(self, x, context=None):
        identity = x[..., self.mask]
        transform = x[..., ~self.mask]
        
        # Get conditioning input
        conditioner_input = transform
        if context is not None:
            conditioner_input = torch.cat([conditioner_input, context], dim=-1)
            
        # Get transformation parameters
        transform_params = self.transform_net(conditioner_input)
        transform_params = transform_params.reshape(
            -1, self.mask.sum().item(), self.num_bins * 3
        )
        
        # Split into widths, heights, and derivatives
        widths = transform_params[..., :self.num_bins]
        heights = transform_params[..., self.num_bins:2*self.num_bins]
        derivatives = transform_params[..., 2*self.num_bins:]
        
        # Apply spline transform
        transformed, log_det = rational_quadratic_spline(
            identity, widths, heights, derivatives
        )
        
        # Merge back transformed and untransformed parts
        output = torch.zeros_like(x)
        output[..., self.mask] = transformed
        output[..., ~self.mask] = transform
        
        return output, log_det
        
    def inverse(self, z, context=None):
        identity = z[..., self.mask]
        transform = z[..., ~self.mask]
        
        # Get conditioning input
        conditioner_input = transform
        if context is not None:
            conditioner_input = torch.cat([conditioner_input, context], dim=-1)
            
        # Get transformation parameters
        transform_params = self.transform_net(conditioner_input)
        transform_params = transform_params.reshape(
            -1, self.mask.sum().item(), self.num_bins * 3
        )
        
        # Split into widths, heights, and derivatives
        widths = transform_params[..., :self.num_bins]
        heights = transform_params[..., self.num_bins:2*self.num_bins]
        derivatives = transform_params[..., 2*self.num_bins:]
        
        # Apply inverse spline transform
        transformed, log_det = rational_quadratic_spline(
            identity, widths, heights, derivatives, inverse=True
        )
        
        # Merge back transformed and untransformed parts
        output = torch.zeros_like(z)
        output[..., self.mask] = transformed
        output[..., ~self.mask] = transform
        
        return output, log_det

class EnhancedRQSFlow(nn.Module):
    def __init__(self, input_dim, num_transforms=5, num_bins=8, hidden_dims=[128, 128],
                 condition_dim=None, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        
        # Create sequence of coupling layers with alternating masks
        self.transforms = nn.ModuleList([
            CouplingLayer(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_bins=num_bins,
                mask_type='alternate' if i % 2 == 0 else 'half',
                activation=activation,
                condition_dim=condition_dim
            ) for i in range(num_transforms)
        ])
        
        # Base distribution
        self.base_dist = torch.distributions.Normal(0, 1)
        
    def forward(self, x, context=None):
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        for transform in self.transforms:
            x, transform_log_det = transform(x, context)
            log_det += transform_log_det
            
        return x, log_det
        
    def inverse(self, z, context=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        
        for transform in reversed(self.transforms):
            z, transform_log_det = transform.inverse(z, context)
            log_det += transform_log_det
            
        return z, log_det
        
    def log_prob(self, x, context=None):
        z, log_det = self.forward(x, context)
        log_base_density = self.base_dist.log_prob(z).sum(dim=-1)
        return log_base_density + log_det
        
    def sample(self, num_samples, context=None, temperature=1.0):
        z = self.base_dist.sample((num_samples, self.input_dim)).to(
            next(self.parameters()).device
        ) * temperature
        x, _ = self.inverse(z, context)
        return x
        
    def sample_with_noise(self, num_samples, context=None, noise_scale=0.1):
        """Sample with added noise for better exploration"""
        samples = self.sample(num_samples, context)
        noise = torch.randn_like(samples) * noise_scale
        return samples + noise


use_gpu = True
gpu_index = 7

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() and use_gpu else 'cpu')
dtype = torch.float64
verbose = True

if __name__ == '__main__':
    # Create a synthetic dataset
    X, _ = make_moons(n_samples=1000, noise=0.0)
    X_test, _ = make_moons(n_samples=1000, noise=0.0)
    X = torch.tensor(X, dtype=dtype)

    # Normalize the data
    X, q1, q2 = normalize_minmax(X)
    X_test, _, _ = normalize_minmax(torch.tensor(X_test, dtype=dtype))
    X_test = X_test.detach().numpy()

    X_train, X_val = split(X, 0.8)
    train_loader, val_loader = create_data_loaders(X_train, X_val, batch_size=512)

    # Create a flow model and load the data
    flowcontainer = FlowContainer(device=device, dtype=dtype, verbose=verbose)
    flowcontainer.load_data(train_loader, val_loader)

    # details of the flow model
    num_dim = X.shape[1]
    num_flow_steps = 2

    transform_type = 'maf'
    
    transform_kwargs = dict(
        hidden_features=256,
        dropout_probability=0.1,
        use_batch_norm=True
    )

    #transform_kwargs = {'hidden_features':256}
    

    flowcontainer.build_flow(num_dims=num_dim, 
                             num_flow_steps=num_flow_steps, 
                             transform_type=transform_type, 
                             transform_kwargs=transform_kwargs
                            )
    
    #flowcontainer.flow = EnhancedRQSFlow(input_dim=num_dim, num_transforms=num_flow_steps, num_bins=8, hidden_dims=[128, 128]).to(device)
    
    # Train the flow model
    train_kwargs = {
        'lambdaL2': 0.0,
        'early_stopping': True,
        'target_distribution': X_test,
        'lr':1e-4
    }

    flowcontainer.train(**train_kwargs)
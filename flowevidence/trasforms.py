# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from nflows.transforms import MaskedAffineAutoregressiveTransform, AffineCouplingTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.splines import rational_quadratic_spline, unconstrained_rational_quadratic_spline
from nflows.nn.nets import MLP, ResidualNet


# Utility functions for rational quadratic spline transformation
def rational_quadratic_spline(inputs, widths, heights, derivatives, inverse=False):
    """Rational Quadratic Spline transformation."""
    num_bins = widths.shape[-1]
    device = inputs.device

    # Normalize the inputs to the (0, 1) range
    inputs = inputs.clamp(1e-5, 1 - 1e-5)

    bin_widths = F.softmax(widths, dim=-1)
    bin_heights = F.softmax(heights, dim=-1)
    bin_derivatives = F.softplus(derivatives)

    cumsum_widths = torch.cumsum(bin_widths, dim=-1)
    cumsum_heights = torch.cumsum(bin_heights, dim=-1)

    cumsum_widths = F.pad(cumsum_widths, pad=(1, 0), mode="constant", value=0.0)
    cumsum_heights = F.pad(cumsum_heights, pad=(1, 0), mode="constant", value=0.0)

    bin_idx = torch.sum(inputs[..., None] > cumsum_widths, dim=-1) - 1

    input_left = cumsum_widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_right = cumsum_widths.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    bin_width = (input_right - input_left).clamp(min=1e-5)

    theta = (inputs - input_left) / bin_width

    # Compute splines
    numerator = (
        bin_heights[..., bin_idx] * (theta ** 2) +
        bin_derivatives[..., bin_idx] * theta * (1 - theta)
    )
    denominator = (
        bin_heights[..., bin_idx + 1] * (1 - theta) ** 2 +
        bin_derivatives[..., bin_idx + 1] * theta * (1 - theta)
    )

    outputs = input_left + bin_width * numerator / denominator

    return outputs

# MLP for parameters of the RQS
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], activation=F.relu):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Masked Rational Quadratic Spline Transform
class MaskedRQSplineTransform(nn.Module):
    def __init__(self, input_dim, num_bins=10, hidden_dims=[128, 128]):
        super().__init__()
        self.num_bins = num_bins
        self.widths = MLP(input_dim, num_bins, hidden_dims)
        self.heights = MLP(input_dim, num_bins, hidden_dims)
        self.derivatives = MLP(input_dim, num_bins + 1, hidden_dims)

    def forward(self, x):
        widths = self.widths(x)
        heights = self.heights(x)
        derivatives = self.derivatives(x)
        return rational_quadratic_spline(x, widths, heights, derivatives)

# Full flow model
class RQSFlow(nn.Module):
    def __init__(self, input_dim, num_transforms=5, num_bins=10, hidden_dims=[128, 128]):
        super().__init__()
        self.transforms = nn.ModuleList([
            MaskedRQSplineTransform(input_dim, num_bins, hidden_dims)
            for _ in range(num_transforms)
        ])

    def forward(self, x):
        log_det_jacobian = 0
        for transform in self.transforms:
            x = transform(x)
        return x

# Define a custom Masked Coupling transform using Rational Quadratic Splines
class MaskedCouplingRQSpline(PiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, features, mask, transform_net_create_fn, num_bins):
        super().__init__(mask, transform_net_create_fn, num_bins=num_bins)

    

def get_transform(model: str = 'maf'):
    """
    Returns the transformation class and its default keyword arguments based on the specified model type.
    
    Args:
        model (str): The type of transformation model to use. Options are:
                    - 'maf': Masked Affine Autoregressive Transform
                    - 'nvp': Affine Coupling Transform
                    - 'rqs': Masked Coupling RQ Spline
                    Default is 'maf'.
    Returns:
        transform_class (Transform): The transformation class to use.
        default_kwargs (dict): The default keyword arguments for the transformation class.    
    """
    
    Transforms_all = {
        'maf': MaskedAffineAutoregressiveTransform,
        'nvp': AffineCouplingTransform,
        'rqs': MaskedCouplingRQSpline,
        'mrqs': MaskedPiecewiseRationalQuadraticAutoregressiveTransform
    }

    maf_kwargs = {
        'hidden_features': 128,
        'dropout_probability': 0.1,
        'use_batch_norm': False
    }

    nvp_kwargs = {
        'mask': None,
        'transform_net_create_fn': lambda in_features, out_features: ResidualNet(
                in_features,
                out_features,
                hidden_features=128,
                num_blocks=2,
                activation=nn.ReLU()
            )
    }

    rqs_kwargs = {
        'mask': None,
        'transform_net_create_fn': lambda in_features, out_features: MLP(
                tuple(in_features),
                tuple(out_features),
                hidden_sizes=[128, 128],
                activation=nn.ReLU(),
                activate_output=True
            ),
        'num_bins': 8
    }

    mrqs_kwargs = {
        'hidden_features': 128,
        'num_bins': 8,
        'num_blocks': 2,
        'dropout_probability': 0.1,
        'use_batch_norm': True
    }

    default_kwargs = {
        'maf': maf_kwargs,
        'nvp': nvp_kwargs,
        'rqs': rqs_kwargs,
        'mrqs': mrqs_kwargs
    }

    return Transforms_all[model], default_kwargs[model]
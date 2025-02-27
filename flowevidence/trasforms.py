import torch
import torch.nn as nn
import normflows as nf

def get_rqs_transform(num_flow_steps, num_dims, **kwargs):
    """
    Returns a Masked Coupling Rational Quadratic Spline transformation.

    Args:
        num_flow_steps (int): The number of flow steps in the transformation.
        num_dims (int): The number of dimensions in the input.
        **kwargs: Additional keyword arguments for the transformation.
    Returns:
        flows (list): A list of the transformation layers. 
    """
    
    flows = []
    for _ in range(num_flow_steps):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_dims, **kwargs)]
        flows += [nf.flows.LULinearPermute(num_dims)]
    return flows

def get_maf_transform(num_flow_steps, num_dims, **kwargs):
    """
    Returns a Masked Affine Autoregressive Transform.

    Args:
        num_flow_steps (int): The number of flow steps in the transformation.
        num_dims (int): The number of dimensions in the input.
        **kwargs: Additional keyword arguments for the transformation.
    Returns:
        flows (list): A list of the transformation layers. 
    """
    
    flows = []
    for _ in range(num_flow_steps):
        # Add MAF layer with configurable neural network
        flows.append(
            nf.flows.MaskedAffineAutoregressive(
                num_dims,
                **kwargs
            )
        )
        
        # Add batch normalization if requested
        if kwargs['use_batch_norm']:
            flows.append(nf.flows.ActNorm(num_dims))
            
        # Add permutation layer to mix features
        flows.append(nf.flows.LULinearPermute(num_dims))
    return flows

def get_nvp_transform(num_flow_steps, num_dims, **kwargs):
    """
    Returns an Affine Coupling Transform.

    Args:
        num_flow_steps (int): The number of flow steps in the transformation.
        num_dims (int): The number of dimensions in the input.
        **kwargs: Additional keyword arguments for the MLP.
    
    Returns:
        flows (list): A list of the transformation layers. 
    """

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(num_dims)])
    if 'layers' in kwargs.keys() and kwargs['layers'] is not None:
        pass
    else:
        hidden_multiplier = kwargs.pop('hidden_multiplier')
        kwargs['layers'] = [num_dims, hidden_multiplier * num_dims, hidden_multiplier * num_dims, num_dims] # Default MLP layers

    flows = []
    for i in range(num_flow_steps):
        s = nf.nets.MLP(**kwargs)
        t = nf.nets.MLP(**kwargs)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(num_dims)]
    
    return flows

def get_flow_builder(model):
    """
    Returns the transformation class based on the specified model type.

    Args:
        model (str): The model type.
    
    Returns:
        get_transform (function): The transformation class to use.
    """

    transform_classes = {
        'maf': get_maf_transform,
        'nvp': get_nvp_transform,
        'rqs': get_rqs_transform
    }

    return transform_classes[model]

def get_tranform_kwargs(model: str):
    """
    Returns the default keyword arguments for the specified model type.

    Args:
        model (str): The model type.
    
    Returns:
        kwargs (dict): The default keyword arguments for the model type.
    """

    maf_dict = {
        'hidden_features': 128,
        'activation': nn.ReLU(),
        'use_batch_norm': True,
        'dropout_probability': 0.1,
    }

    nvp_dict = {
        'layers': None,
        'leaky': 0.0,
        'init_zeros': True,
        'hidden_multiplier': 2,
        'dropout': 0.0,
    }

    rqs_dict = {
        'num_blocks': 5,
        'num_hidden_channels': 128,
        'num_bins': 8,
        'dropout_probability': 0.1,
    }

    kwargs = {
        'maf': maf_dict,
        'nvp': nvp_dict,
        'rqs': rqs_dict
    }

    return kwargs[model]

def get_flow_transforms(num_dims: int, 
                        num_flow_steps: int, 
                        model: str = 'maf', 
                        extra_kwargs: dict = {}):
    """
    Returns the transformation class and its default keyword arguments based on the specified model type.
    
    Args:
        num_dims (int): The number of dimensions in the input.
        num_flow_steps (int): The number of flow steps in the transformation.
        model (str): The model type.
        extra_kwargs (dict): Additional keyword arguments for the transformation.
        
    Returns:
        transform_class (Transform): The transformation class to use.
        default_kwargs (dict): The default keyword arguments for the transformation class.    
    """
    
    default_kwargs = get_tranform_kwargs(model).copy()
    default_kwargs.update(extra_kwargs) 

    build_flow = get_flow_builder(model)

    flows = build_flow(num_flow_steps, num_dims, **default_kwargs)

    return flows
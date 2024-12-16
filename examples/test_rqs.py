import torch
import torch.nn as nn

from flowevidence.core import FlowContainer
from flowevidence.utils import split, create_data_loaders, normalize_minmax, denormalize_minmax
from sklearn.datasets import make_moons

import normflows as nf

import numpy as np
import matplotlib.pyplot as plt

import pysco
pysco.plot.default_plotting()

use_gpu = True
gpu_index = 7

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() and use_gpu else 'cpu')
dtype = torch.float64
verbose = True

if __name__ == '__main__':
    # Create a synthetic dataset
    X, _ = make_moons(n_samples=10000, noise=0.1)
    X_test, _ = make_moons(n_samples=10000, noise=0.1)
    X = torch.tensor(X, dtype=dtype)

    # Normalize the data
    X, q1, q2 = normalize_minmax(X)
    X_test, _, _ = normalize_minmax(torch.tensor(X_test, dtype=dtype))
    X_test = X_test.detach().numpy()

    X_train, X_val = split(X, 0.8)
    train_loader, val_loader = create_data_loaders(X_train, X_val, batch_size=1000)

    # Create a flow model and load the data
    flowcontainer = FlowContainer(device=device, dtype=dtype, verbose=verbose)
    flowcontainer.load_data(train_loader, val_loader)

    # details of the flow model
    num_dim = X.shape[1]
    num_flow_steps = 10

    torch.manual_seed(0)

    flowcontainer.build_flow(
        num_dims=num_dim,
        num_flow_steps=num_flow_steps,
        transform_type='rqs',
        transform_kwargs={
            
        }
    )

    # Train the flow model
    train_kwargs = {
        'epochs': 5000,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'lambdaL2': 0.0,
        'early_stopping': True,
        'target_distribution': X_test,
    }

    flowcontainer.train(**train_kwargs)
# coding: utf-8

import os
import torch
from torch.utils.data import DataLoader

from typing import Optional, Callable
import logging
import numpy as np
import h5py
from .utils import *
from tqdm import tqdm

try:
    from eryn.backends import HDFBackend
    from eryn.utils import get_integrated_act
    eryn_here = True
except (ImportError, ModuleNotFoundError):
    logging.warning("Eryn is not installed. Please install Eryn to use the ErynEvidenceFlow class.")
    eryn_here = False

try:
    from pysco.eryn import SamplesLoader
    pysco_here = True
except (ImportError, ModuleNotFoundError):
    logging.warning("Pysco is not installed. Please install Pysco to use the ErynEvidenceFlow class.")
    pysco_here = False

__all__ = ['FlowContainer', 'EvidenceFlow', 'ErynEvidenceFlow']

class FlowContainer:
    """
    A container for managing and training a flow-based model.
    
    Args:
        device (Union[str, torch.device]): Device to run the model on. Default is 'cpu'.
        dtype (torch.dtype): Data type for tensors. Default is torch.float64.
        verbose (bool): Whether to print verbose output during training. Default is False.
        
    Methods:
        build_flow(num_dims=None):
            Builds the flow model using the specified parameters.
        load_data(train_loader, val_loader=None):
            Loads the training and validation data loaders.
        train(start_epoch=0, epochs=1000, lr=1e-3, lambdaL2=None, path='./', filename='trainedflow.pth', target_distribution=None):
            Trains the flow model with the specified parameters.
        load(path='./', filename='trainedflow.pth'):
            Loads a trained flow model from the specified path.
    """
    
    def __init__(self, 
                device: str | torch.device = 'cpu',
                dtype: torch.dtype = torch.float64,
                verbose: bool = False,
                ):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)
        setup_logging(verbose)
        self.verbose = verbose
    
        self.train_loader = None
        self.val_loader = None

    def build_flow(self, 
                   num_dims: int,
                   num_flow_steps: int = 16,
                   transform_type: str = 'nvp',
                   transform_kwargs: dict = {},
                   ):
        """
        Builds the flow model using the specified parameters.
        This method initializes the flow model by calling the `get_flow` function with the 
        number of dimensions, number of flow steps, type of transformation, and device to be used for computation.

        Args:  
            num_dims (int): The number of dimensions for the flow model.
            num_flow_steps (int): The number of flow steps in the model. Default is 16.
            transform_type (str): The type of transformation to use. Default is 'nvp'.
            transform_kwargs (dict): Additional keyword arguments for the transformation. Default is {}.
        """

        self.flow = get_flow(num_dims, 
                            num_flow_steps=num_flow_steps, 
                            transform_type=transform_type,
                            transform_kwargs=transform_kwargs,
                            device=self.device
                            )

    def load_data(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Loads the training and validation data loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation data loader. Default is None.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, 
              start_epoch: int = 0, 
              epochs: int = 1000, 
              lr: float = 1e-3, 
              weight_decay: float = 0.0,
              lambdaL2: Optional[float] = None,
              early_stopping: bool | Callable = False,
              stopping_kwargs: Optional[dict] = {},
              path: str = './', 
              filename: str = 'trainedflow.pth', 
              target_distribution: Optional[np.ndarray] = None
              ):
        """
        Train the flow model.

        Args:
            start_epoch (int, optional): The starting epoch for training. Defaults to 0.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1000.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.0.
            lambdaL2 (Optional[float], optional): The L2 regularization parameter. Defaults to None.
            early_stopping (Optional[bool], optional): Whether to use early stopping. Defaults to False.
            stopping_kwargs (Optional[dict], optional): Keyword arguments for early stopping. Defaults to {}.
            path (str, optional): The path to save the trained model and diagnostics. Defaults to './'.
            filename (str, optional): The filename for the saved model. Defaults to 'trainedflow.pth'.
            target_distribution (Optional[np.ndarray], optional): The target distribution for diagnostics. Defaults to None.
        """

        logging.info("Training flow for {} epochs".format(epochs - start_epoch))
        
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr, weight_decay=weight_decay)
        if self.val_loader:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                factor=0.5,
                                                                patience=100,
                                                                threshold=1e-4)
                                                                
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        current_lr = lr

        epochs_losses = []
        train_losses = []
        val_losses = []

        stopping_fn = None
        converged = False
        if isinstance(early_stopping, bool) and early_stopping:
            stopping_fn = EarlyStopping(**stopping_kwargs)
        elif isinstance(early_stopping, Callable):
            stopping_fn = early_stopping
        
        else:
            logging.info("Early stopping disabled")
            
        trainedpath = path + filename
        savepath = path + "diagnostic/"
        os.makedirs(savepath, exist_ok=True)

        logging.info("Training started")
        logging.info(f"Saving diagnostics to {savepath}")

        if epochs < start_epoch:
            logging.info("Resuming training")
            epochs = start_epoch + epochs

        epoch_iterator = tqdm(range(start_epoch, epochs), desc="Training", disable=not self.verbose)

        for epoch in epoch_iterator:
            train_loss = self._train_one_epoch(optimizer, lambdaL2)
            val_loss = self._validate_one_epoch(lambdaL2) if self.val_loader else None
            scheduler.step(val_loss) if self.val_loader else scheduler.step()


            if stopping_fn:
                if stopping_fn(val_loss):
                    logging.info(f"Early stopping at epoch {epoch}")
                    converged = True
                    break

            if epoch  > 0 and epoch % 100 == 0:
                if self.verbose:
                    self._log_epoch(epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, target_distribution, savepath)
                    if scheduler.get_last_lr()[0] != current_lr:
                        current_lr = scheduler.get_last_lr()[0]
                        logging.info(f"New learning rate: {scheduler.get_last_lr()[0]}")
                    logging.info("Saving model @ epoch {}".format(epoch))

                self._save_model(epoch, optimizer, scheduler, trainedpath)

        if stopping_fn and not converged:
            logging.warning("Early stopping did not trigger")

        self._save_model(epochs, optimizer, scheduler, trainedpath)
        logging.debug("Training finished")
        
        logging.debug("Saving diagnostics")
        epochs_losses.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        self._save_diagnostics(epochs_losses, train_losses, val_losses, target_distribution, savepath)

    def _train_one_epoch(self, optimizer, lambdaL2):
        """
        Train the flow model for one epoch.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            lambdaL2 (Optional[float]): The L2 regularization parameter. Defaults to None.
        
        Returns:
            float: The average training loss for the epoch.
        """

        self.flow.train()
        train_loss = 0
        Nbatches = 0 #number of batches that are not nan or inf
        for batch in self.train_loader:
            batch = batch[0].to(self.device, non_blocking=self.device.type == 'cuda')

            # Check if any samples are at the boundary
            at_boundary = torch.any(torch.abs(batch) > 0.999, dim=1)
            if torch.any(at_boundary):
                # Apply small jitter to boundary points to avoid numerical issues
                batch = batch + torch.randn_like(batch) * 1e-4
                
            optimizer.zero_grad()
            #breakpoint()
            #loss = loss_fn(self.flow, batch)
            loss = self.flow.forward_kld(batch)
            l2_reg = l2_regularization(self.flow, lambdaL2) if lambdaL2 else 0
            loss = loss + l2_reg
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                Nbatches += 1
        return train_loss / max(1, Nbatches)

    def _validate_one_epoch(self, lambdaL2):
        """
        Validate the flow model for one epoch.

        Args:
            lambdaL2 (Optional[float]): The L2 regularization parameter. Defaults to None.

        Returns:
            float: The average validation loss for the epoch.
        """

        self.flow.eval()
        val_loss = 0
        Nbatches = 0 #number of batches that are not nan or inf
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch[0].to(self.device, non_blocking=self.device.type == 'cuda')
                loss = self.flow.forward_kld(batch) #loss_fn(self.flow, batch)
                l2_reg = l2_regularization(self.flow, lambdaL2) if lambdaL2 else 0
                loss = loss + l2_reg
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    val_loss += loss.item()
                    Nbatches += 1
                # else:
                #     breakpoint()
        return val_loss / max(1, Nbatches)

    def _log_epoch(self, epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, target_distribution, savepath):
        """
        Log the training and validation losses for the epoch.

        Args:
            epoch (int): The current epoch.
            train_loss (float): The training loss for the epoch.
            val_loss (float): The validation loss for the epoch.
            epochs_losses (list): List of epochs.
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
            target_distribution (np.ndarray): The target distribution for diagnostics.
            savepath (str): The path to save the diagnostics.
        """

        if val_loss is not None:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        else:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}')

        epochs_losses.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        self._save_diagnostics(epochs_losses, train_losses, val_losses, target_distribution, savepath)

    def _save_model(self, epochs, optimizer, scheduler, trainedpath):
        """
        Save the trained model.

        Args:
            epochs (int): The number of epochs trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            trainedpath (str): The path to save the trained model.
        """

        savedict = {
            'epoch': epochs,
            'model_state_dict': self.flow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(savedict, trainedpath)

    def _save_diagnostics(self, epochs_losses, train_losses, val_losses, target_distribution, savepath, ndim=10):
        """
        Save diagnostics for the trained model.

        Args:
            epochs_losses (list): List of epochs.
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
            target_distribution (np.ndarray): The target distribution for diagnostics.
            savepath (str): The path to save the diagnostics.
            ndim (int, optional): The number of dimensions to plot. Defaults to 10.
        """
        Nsamples_default = int(1e4)
        Nsamples = target_distribution.shape[0] if target_distribution is not None else Nsamples_default
        Nsamples = min(Nsamples, Nsamples_default)

        samples_here, log_prob = self.flow.sample(Nsamples)
        samples_here = samples_here.cpu().detach().numpy()

        lossplot(epochs_losses, train_losses, val_losses, plot_dir=savepath, savename='flow_loss')
        try:
            cornerplot_training(samples_here[:, :ndim], target_distribution[:, :ndim], epoch=epochs_losses[-1], plot_dir=savepath, savename=f'flow_cornerplot')
        except Exception as e:
            logging.error(f"Error plotting cornerplot: {e}")

        logging.debug("Diagnostics saved")

    def load(self, path: str = './', filename: str = 'trainedflow.pth'):
        """
        Load a trained flow model from the specified path.

        Args:
            path (str, optional): The path to the saved model. Defaults to './'.
            filename (str, optional): The filename of the saved model. Defaults to 'trainedflow.pth'.
        """

        try:
            loadpath = path + filename
            logging.debug(f"Loading flow from {loadpath}")
            checkpoint = torch.load(loadpath)
            self.flow.load_state_dict(checkpoint['model_state_dict'])
            logging.debug("Flow loaded")
            return True
        except Exception as e:
            logging.error(f"Error loading flow: {e}")
            return False

class EvidenceFlow(FlowContainer):
    """
    A class for computing the log evidence (logZ) using a trained flow model and the posterior values associated with MCMC samples.

    Args:
        posterior_samples (np.ndarray or dict): The posterior samples to use for training the flow model. If a dictionary, the values are concatenated along the last axis.
        logposterior_values (np.ndarray): The log posterior values associated with the posterior samples.
        num_flow_steps (int): Number of flow steps in the model. Default is 16.
        transform_type (str): The type of transformation to use. Default is 'nvp'.
        transform_kwargs (dict): Additional keyword arguments for the transformation. Default is {}.
        device (str or torch.device): Device to run the model on. Default is 'cpu'.
        verbose (bool): Whether to print verbose output during training. Default is False.
        dtype (torch.dtype): Data type for tensors. Default is torch.float64.
        Nbatches (int): Number of batches. Default is 1.
        split_ratio (float): Ratio to split data into training and validation sets. Default is 0.8.
        conversion_method (str): Method for data conversion to the flow latent space ('normalize_minmax' or 'normalize_gaussian'). 
                                 Default is 'normalize_minmax'.    
        autoencoder (nn.Module): An autoencoder to encode the training and validation samples. Default is None.
        train_autoencoder_kwargs (dict): Keyword arguments for training the autoencoder. Default is {}.
    
    Methods:
        _setup_conversions(conversion_method):
            Sets up the conversion methods to the latent space.
        _process_posterior_samples(posterior_samples):
            Processes the posterior samples and converts them to tensors.
        _process_tensors():
            Processes tensors, shuffles samples, splits data, and creates data loaders.
        get_logZ(load_kwargs={}, train_kwargs={}):
            Computes the log evidence (logZ) by building and training the flow model if necessary.
    """

    def __init__(self, 
                 posterior_samples: np.ndarray | dict = None,
                 logposterior_values: np.ndarray = None,
                 num_flow_steps: int = 16, 
                 transform_type: str = 'nvp',
                 transform_kwargs: dict = {},
                 device: str | torch.device = 'cpu', 
                 verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 Nbatches: int = 1,
                 split_ratio: float = 0.8,
                 conversion_method: str = 'minmax',
                 autoencoder: nn.Module = None,
                 train_autoencoder_kwargs: dict = {},
                 ):
        
        super().__init__(device, dtype, verbose)

        self.num_flow_steps = num_flow_steps
        self.transform_type = transform_type
        self.transform_kwargs = transform_kwargs
        
        self.split_ratio = split_ratio
        self._setup_conversions(conversion_method)
        self.posterior_samples = self._process_posterior_samples(posterior_samples)

        self.Nsamples, self.num_dims_full = self.posterior_samples.shape
        self.Nbatches = Nbatches if Nbatches < self.Nsamples else self.Nsamples
        self.batch_size = self.Nsamples // self.Nbatches
        self.logposterior_values = logposterior_values

        # Autoencoder
        self.autoencoder = autoencoder
        self.train_autoencoder_kwargs = train_autoencoder_kwargs

        self._process_tensors()
        self.num_dims = self.latent_target.shape[1]

    def _setup_conversions(self, conversion_method):
        """
        Sets up the conversion methods for transforming data to and from latent space.
        
        Args:
            conversion_method (str): The method to use for conversion. Must be one of 'normalize_minmax' or 'normalize_gaussian'.
        
        Raises:
            ValueError: If an invalid conversion method is provided.
        """

        allowed_methods = {
        'minmax': (normalize_minmax, denormalize_minmax),
        'gaussian': (normalize_gaussian, normalize_gaussian),
        'sigmoid': (normalize_sigmoid, denormalize_sigmoid),
        #'logit': (normalize_logit, denormalize_logit)
    }

        if conversion_method in allowed_methods:
            self._to_latent_space, self._from_latent_space = allowed_methods[conversion_method]
        else:
            raise ValueError(f"Invalid conversion method: {conversion_method}. Choose from {list(allowed_methods.keys())}")

    def _process_posterior_samples(self, posterior_samples):
        """
        Processes posterior samples by concatenating them if they are in dictionary form 
        and converting them to a PyTorch tensor.
        
        Args:
            posterior_samples (dict or array-like): The posterior samples to process. 
                If a dictionary, the values are concatenated along the last axis.
        
        Returns:
            torch.Tensor: The processed posterior samples as a PyTorch tensor.
        """

        if isinstance(posterior_samples, dict):
            posterior_samples = np.concatenate([posterior_samples[key] for key in posterior_samples.keys()], axis=-1)
        posterior_samples = torch.tensor(posterior_samples, dtype=self.dtype)
        return posterior_samples
    
    def _process_tensors(self):
        """
        Processes the posterior samples by converting them to latent space, shuffling, 
        splitting into training and validation sets, and creating data loaders.
        This method performs the following steps:
        1. Converts posterior samples to latent space.
        2. Shuffles the latent samples.
        3. Splits the shuffled samples into training and validation sets based on the split ratio.
        4. Creates data loaders for the training and validation sets.
        5. Loads the data using the created data loaders.
        6. If an autoencoder is provided, it is used to encode the training and validation samples.        

        Attributes:
            self.q1: The first component of the latent space representation.
            self.q2: The second component of the latent space representation.
            self.latent_target: The latent space representation of the posterior samples as a NumPy array.
        """
        
        latent_samples, self.q1, self.q2 = self._to_latent_space(self.posterior_samples)
        self.latent_target = latent_samples.to(self.device)
        #breakpoint()

        shuffled_samples = shuffle(latent_samples)
        if self.split_ratio:
            train_samples, val_samples = split(shuffled_samples, self.split_ratio)
        else:
            train_samples, val_samples = shuffled_samples, None

        train_loader, val_loader = create_data_loaders(train_samples, val_samples, batch_size=self.batch_size, num_workers=0, pin_memory=self.device.type == 'cuda')
        
        if self.autoencoder:
            self._sanity_check_autoencoder(train_loader, val_loader, self.train_autoencoder_kwargs)
            train_samples, val_samples = self._encode_tensors(train_samples, val_samples)
            train_loader, val_loader = create_data_loaders(train_samples, val_samples, batch_size=self.batch_size, num_workers=0, pin_memory=self.device.type == 'cuda')

        self.load_data(train_loader, val_loader)

    def _sanity_check_autoencoder(self, 
                                  train_loader: DataLoader, 
                                  val_loader: DataLoader,
                                  training_kwargs: dict
                                  ):
        """
        Checks if the autoencoder is trained and trains it if necessary.

        Args:
            train_loader (DataLoader): The training samples to train the autoencoder on.
            val_loader (DataLoader): The validation samples to train the autoencoder on.
            training_kwargs (dict): Keyword arguments for training the autoencoder.
        """

        filename = self.train_autoencoder_kwargs.get('filename', 'autoencoder.pth')
        if os.path.exists(training_kwargs['path'] + filename):
            logging.info("Loading autoencoder")
            self.autoencoder.load_model(path=training_kwargs['path'] + filename)
        else:
            logging.info("Autoencoder not found")

        if self.autoencoder.trained:
            # check if autoencoder is trained. If not, train it
            logging.info("Autoencoder already trained")
        else:
            logging.info("Training autoencoder")
            self.autoencoder.train(train_loader, val_loader, **training_kwargs)
            logging.info("Autoencoder trained")


    def _encode_tensors(self, 
                        train_tensor: torch.Tensor, 
                        val_tensor: Optional[torch.Tensor] = None
                        ):
        """
        Encodes the training and validation samples using the autoencoder. If the autoencoder is not trained, it will be trained. If the autoencoder is not provided, the samples are returned as is.
        
        Args:
            train_loader (DataLoader): The training samples to train the autoencoder on.
            val_loader (DataLoader): The validation samples to train the autoencoder on.
            training_kwargs (dict): Keyword arguments for training the autoencoder.
        """
        train_tensor = self.autoencoder.encode(train_tensor.to(self.device)).to('cpu')
        if val_tensor is not None:
            val_tensor = self.autoencoder.encode(val_tensor.to(self.device)).to('cpu')
        self.latent_target = self.autoencoder.encode(self.latent_target)
        
        return train_tensor, val_tensor
            

    def get_logZ(self,
                 load_kwargs: dict = {},
                 train_kwargs: dict = {},
                 ): 
        """
        Computes the log evidence (logZ) by building and training the flow model if necessary.

        Args:
            load_kwargs (dict): Keyword arguments for loading the flow model.
            train_kwargs (dict): Keyword arguments for training the flow model.
        
        Returns:
            logZ (float): The mean log evidence.
            dlogZ (float): The standard deviation of the log evidence.
        """
        self._sanity_check_flow(load_kwargs, train_kwargs)
        
        logProb = self.flow.log_prob(self.latent_target).cpu().detach().numpy()
        logZ = self.logposterior_values - logProb
        mean, std = np.mean(logZ), np.std(logZ)
        logging.debug(f"LogZ: {mean} +/- {std}")
        
        return mean, std
    
    def get_draws(self, 
                  load_kwargs: dict = {}, 
                  train_kwargs: dict = {}, 
                  num_draws: int = 10000
                  ):
        """
        Draw samples from the trained flow model. If no model is loaded or trained, it will be trained.

        Args:
            load_kwargs (dict): Keyword arguments for loading the flow model. Refer to the documentation for the `load` method.
            train_kwargs (dict): Keyword arguments for training the flow model. Refer to the documentation for the `train` method.
            num_draws (int, optional): The number of samples to draw. Defaults to 10000.
        
        Returns:
            samples (np.ndarray): The drawn samples transformed in the original space.
        """
        self._sanity_check_flow(load_kwargs, train_kwargs)
        samples, log_prob = self.flow.sample(num_draws).cpu().detach().numpy()

        if self.autoencoder:
            samples = self.autoencoder.decode(samples)

        converted = self._from_latent_space(samples, self.q1, self.q2)

        return converted
    
    def _sanity_check_flow(self, 
                           load_kwargs: dict = {}, 
                           train_kwargs: dict = {}
                           ):
        """
        Checks if the flow model is loaded or trained and loads or trains it if necessary.
        
        Args:
            load_kwargs (dict): Keyword arguments for loading the flow model. Refer to the documentation for the `load` method.
            train_kwargs (dict): Keyword arguments for training the flow model. Refer to the documentation for the `train` method.
        """
        if not hasattr(self, 'flow'):
            logging.info("Building flow")
            self.build_flow(self.num_dims, self.num_flow_steps, self.transform_type, self.transform_kwargs)
        
        load = self.load(**load_kwargs)
        train_kwargs_here = train_kwargs.copy()
        
        if not load:
            train_kwargs_here['target_distribution'] = self.latent_target.cpu().detach().numpy()
            if 'path' not in train_kwargs_here and 'path' in load_kwargs:
                train_kwargs_here['path'] = load_kwargs['path']

            if 'filename' not in train_kwargs_here and 'filename' in load_kwargs:
                train_kwargs_here['filename'] = load_kwargs['filename']

            logging.info("Training flow")
            self.train(**train_kwargs_here)

class ErynEvidenceFlow(EvidenceFlow):
    """
    Wrapper class for using the ``EvidenceFlow`` class directly with a backend from the ``Eryn`` mcmc sampler. 
    It stores the samples and logP values in a file for faster loading.
    
    Args:
        backend (str or HDFBackend): The backend to load the samples from.
        loader (SamplesLoader): A pysco.eryn.SamplesLoader object to load the samples from.
        samples_file (str): The file to save the samples and logP values to. Default is './samples.h5'.
        ess (int): The effective sample size. Default is 1e4. It is used to compute the number of samples to discard and thin if they are `None`.
        discard (int): The number of samples to discard. Default is None.
        thin (int): The thinning factor. Default is None.
        leaves_to_ndim (bool): Whether to reshape the leaves to ndim. Default is False.
        num_flow_steps (int): Number of flow steps in the model. Default is 16.
        transform_type (str): The type of transformation to use. Default is 'nvp'.
        transform_kwargs (dict): Additional keyword arguments for the transformation. Default is {}.
        device (str or torch.device): Device to run the model on. Default is 'cpu'.
        verbose (bool): Whether to print verbose output during training. Default is False.
        dtype (torch.dtype): Data type for tensors. Default is torch.float64.
        Nbatches (int): Number of batches. Default is 1.
        split_ratio (float): Ratio to split data into training and validation sets. Default is 0.8.
        conversion_method (str): Method for data conversion to the flow latent space ('normalize_minmax' or 'normalize_gaussian'). Default is 'normalize_minmax'.
        autoencoder (nn.Module): An autoencoder to encode the training and validation samples. Default is None.
        train_autoencoder_kwargs (dict): Keyword arguments for training the autoencoder. Default is {}.
    """

    def __init__(self,
                backend: str | HDFBackend = None,
                loader: SamplesLoader = None,
                samples_file: h5py.File = './samples.h5',
                ess: int = int(1e4),
                discard: int = None,
                thin: int = None,
                leaves_to_ndim: bool = False,
                num_flow_steps: int = 16, 
                transform_type: str = 'nvp',
                transform_kwargs: dict = {},
                device: str | torch.device = 'cpu', 
                verbose: bool = False,
                dtype: torch.dtype = torch.float64,
                Nbatches: int = 1,
                split_ratio: float = 0.8,
                conversion_method: str = 'normalize_minmax',
                autoencoder: nn.Module = None,
                train_autoencoder_kwargs: dict = {}):
        
        if not eryn_here:
            raise ImportError("Eryn is not installed. Please install Eryn to use this class, or \
                              use the EvidenceFlow class instead.")
        
        if os.path.exists(samples_file):
            with h5py.File(samples_file, 'r') as f:
                results = f['results']
                samples_group = results['samples']
                samples = {}
                for key in samples_group.keys():
                    samples[key] = samples_group[key][:]
                logP = results['logP'][:]
        else:
            if backend is None and loader is None:
                raise ValueError("Either a backend or a loader must be provided.")
        
            elif loader is None and backend is not None:
                if pysco_here:
                    loader = SamplesLoader(backend)
                    samples, logL, logP = loader.load(ess=ess, discard=discard, thin=thin, squeeze=False, leaves_to_ndim=leaves_to_ndim)
                else:
                    if isinstance(backend, str):
                        backend = HDFBackend(backend)

                    samples, logP = self._load_samples_posterior(backend, ess, leaves_to_ndim=leaves_to_ndim)
            
            else:
                samples, logL, logP = loader.load(ess=ess, discard=discard, thin=thin, squeeze=False, leaves_to_ndim=leaves_to_ndim)

            # Save the samples and logP to a file
            os.makedirs(os.path.dirname(samples_file), exist_ok=True)
            with h5py.File(samples_file, 'w') as f:
                g = f.create_group('results')
                chain = g.create_group('samples')
                for key in samples.keys():
                    chain.create_dataset(key, data=samples[key])
                g.create_dataset('logP', data=logP)

        super().__init__(posterior_samples=samples, 
                         logposterior_values=logP, 
                         num_flow_steps=num_flow_steps, 
                         transform_type=transform_type,
                         transform_kwargs=transform_kwargs, 
                         device=device, 
                         verbose=verbose, 
                         dtype=dtype, 
                         Nbatches=Nbatches, 
                         split_ratio=split_ratio, 
                         conversion_method=conversion_method,
                         autoencoder=autoencoder,
                         train_autoencoder_kwargs=train_autoencoder_kwargs
                         )
        
    def _compute_discard_thin(self, 
                              samples: dict, 
                              ess: int = int(1e4)
                              ):
        """
        Compute the number of samples to discard and thin. Snippet adapted from from: `https://github.com/asantini29/pysco`

        Args:
            ess (int): Effective sample size. Default is 1e4.
        
        Returns:    
            discard (int): The number of samples to discard.
            thin (int): The thinning factor.
        """

        tau = {}
        for name in samples.keys():
            chain = samples[name]
            nsteps, ntemps, nw, nleaves, ndims = chain.shape
            chain = chain.reshape(nsteps, ntemps, nw, nleaves * ndims)
            tau[name] = get_integrated_act(chain, average=True)
        
        taus_all = []

        for name in tau.keys():
            tau_here = np.max(tau[name])
            if np.isfinite(tau_here):
                taus_all.append(tau_here)
        
        thin = int(np.max(taus_all))
        print("Number of steps: ", nsteps)

        ess = int(ess)
        N_keep = int(np.ceil(ess * self.thin / nw))
        print("Number of samples to keep: ", N_keep)
        discard = max(5000, self.backend.iteration - N_keep)

        return discard, thin
    
    def _load_samples_posterior(self, 
                                backend: HDFBackend, 
                                ess: int = None, 
                                leaves_to_ndim: bool = False
                                ):
        """
        Load the samples from the backend. If the effective sample size is provided, the number of samples to discard and thin is computed.
        This is NOT compatible with reversible jump MCMC yet.

        Args:
            backend (HDFBackend): The backend to load the samples from.
            ess (int, optional): The effective sample size. Defaults to None.
            leaves_to_ndim (bool, optional): Whether to reshape the leaves to ndim. Defaults to False.

        Returns:
            samples_out (dict): The samples.
            logP (np.ndarray): The log posterior values.
        """

        samples = backend.get_chain()
        samples_out = {}
        if ess:
            discard, thin = self._compute_discard_thin(samples, ess)
        else:
            discard, thin = 0, 1
        
        for name in samples.keys():
            ns, nt, nw, nl, nd = samples[name].shape
            if leaves_to_ndim:
                samples_out[name] = np.squeeze(samples[name][discard::thin, 0]).reshape(-1, nl*nd) #take the first temperature chain and flatten the rest
            else:
                samples_out[name] = np.squeeze(samples[name][discard::thin, 0]).reshape(-1, nd) #take the first temperature chain and flatten the rest
        
        logP = backend.get_log_posterior(discard=discard, thin=thin)[:, 0].flatten()

        return samples_out, logP
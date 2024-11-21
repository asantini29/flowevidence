import torch
from torch.utils.data import DataLoader

from typing import Union, Optional
import logging
import numpy as np

from .utils import *

try:
    from eryn.backends import HDFBackend
    from eryn.utils import get_integrated_act
    eryn_here = True
except (ImportError, ModuleNotFoundError):
    eryn_here = False

try:
    import pysco
    pysco_here = True
except (ImportError, ModuleNotFoundError):
    pysco_here = False

__all__ = ['FlowContainer', 'EvidenceFlow', 'ErynEvidenceFlow']

class FlowContainer:
    """
    A container for managing and training a flow-based model.
    
    Args:
        num_dims (int): Number of dimensions for the flow model.
        num_flow_steps (int): Number of flow steps in the model. Default is 5.
        use_nvp (bool): Whether to use RealNVP architecture. Default is False.
        device (Union[str, torch.device]): Device to run the model on. Default is 'cpu'.
        dtype (torch.dtype): Data type for tensors. Default is torch.float64.
        verbose (bool): Whether to print verbose output during training. Default is False.

    Methods:
        build_flow():
            Builds the flow model using the specified parameters.
        load_data(train_loader, val_loader=None):
            Loads the training and validation data loaders.
        train(start_epoch=0, epochs=1000, lr=1e-3, lambdaL2=None, path='./', filename='trainedflow.pth', target_distribution=None):
            Trains the flow model with the specified parameters.
        load(path='./', filename='trainedflow.pth'):
            Loads a trained flow model from the specified path.
    """
    
    def __init__(self, 
                num_dims: int, 
                num_flow_steps: int = 5, 
                use_nvp: bool = False,
                device: str | torch.device = 'cpu',
                dtype: torch.dtype = torch.float64,
                verbose: bool = False
                ):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)
        setup_logging(verbose)
        self.verbose = verbose

        self.num_dims = num_dims
        self.num_flow_steps = num_flow_steps
        self.use_nvp = use_nvp

        self.flow = None
        self.train_loader = None
        self.val_loader = None

    def build_flow(self):
        """
        Builds the flow model using the specified parameters.
        This method initializes the flow model by calling the `get_flow` function with the 
        number of dimensions, number of flow steps, whether to use NVP (Non-volume Preserving) 
        transformations, and the device to be used for computation.
        """

        self.flow = get_flow(self.num_dims, self.num_flow_steps, use_nvp=self.use_nvp, device=self.device)

    def load_data(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Loads the training and validation data loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation data loader. Default is None.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, start_epoch: int = 0, epochs: int = 1000, lr: float = 1e-3, lambdaL2: Optional[float] = None, path: str = './', filename: str = 'trainedflow.pth', target_distribution: Optional[np.ndarray] = None):
        """
        Train the flow model.

        Args:
            start_epoch (int, optional): The starting epoch for training. Defaults to 0.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1000.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            lambdaL2 (Optional[float], optional): The L2 regularization parameter. Defaults to None.
            path (str, optional): The path to save the trained model and diagnostics. Defaults to './'.
            filename (str, optional): The filename for the saved model. Defaults to 'trainedflow.pth'.
            target_distribution (Optional[np.ndarray], optional): The target distribution for diagnostics. Defaults to None.
        """
        
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        epochs_losses = []
        train_losses = []
        val_losses = []

        trainedpath = path + filename
        savepath = path + "diagnostic/"

        logging.debug("Training started")
        logging.debug(f"Saving diagnostics to {savepath}")

        if epochs < start_epoch:
            logging.debug("Resuming training")
            epochs = start_epoch + epochs

        for epoch in range(start_epoch, epochs):
            train_loss = self._train_one_epoch(optimizer, lambdaL2)
            scheduler.step()
            val_loss = self._validate_one_epoch(lambdaL2) if self.val_loader else None

            if self.verbose and epoch % 100 == 0:
                self._log_epoch(epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, target_distribution, savepath)

        self._save_model(epochs, optimizer, scheduler, trainedpath)
        self._save_diagnostics(epochs_losses, train_losses, val_losses, target_distribution, savepath)

    def _train_one_epoch(self, optimizer, lambdaL2):
        self.flow.train()
        train_loss = 0
        for batch in self.train_loader:
            batch = batch[0].to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(self.flow, batch)
            l2_reg = l2_regularization(self.flow, lambdaL2) if lambdaL2 else 0
            loss = loss + l2_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def _validate_one_epoch(self, lambdaL2):
        self.flow.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch[0].to(self.device)
                loss = loss_fn(self.flow, batch)
                l2_reg = l2_regularization(self.flow, lambdaL2) if lambdaL2 else 0
                loss = loss + l2_reg
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def _log_epoch(self, epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, target_distribution, savepath):
        if val_loss is not None:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        else:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}')

        epochs_losses.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        samples_here = self.flow.sample(10000).cpu().detach().numpy()
        cornerplot_training(samples_here, target_distribution, epoch=epoch, plot_dir=savepath, savename=f'cornerplot_epoch_{epoch}')
        lossplot(epochs_losses, train_losses, val_losses, plot_dir=savepath, savename='losses')

    def _save_model(self, epochs, optimizer, scheduler, trainedpath):
        savedict = {
            'epoch': epochs,
            'model_state_dict': self.flow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(savedict, trainedpath)

    def _save_diagnostics(self, epochs_losses, train_losses, val_losses, target_distribution, savepath):
        logging.debug("Training finished")
        logging.debug("Saving diagnostics")
        if self.verbose:
            samples_here = self.flow.sample(10000).cpu().detach().numpy()
            cornerplot_training(samples_here, target_distribution, epoch=epochs_losses[-1], plot_dir=savepath, savename=f'cornerplot_epoch_{epochs_losses[-1]}')
            lossplot(epochs_losses, train_losses, val_losses, plot_dir=savepath, savename='losses')
            logging.debug("Diagnostics saved")

    def load(self, path: str = './', filename: str = 'trainedflow.pth'):
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
        num_flow_steps (int): Number of flow steps. Defaults to 5.
        use_nvp (bool, optional): Whether to use NVP (Neural Variational Processes). Defaults to False.
        device (str or torch.device, optional): Device to use for computation. Defaults to 'cpu'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        posterior_samples (np.ndarray or dict, optional): Posterior samples. Defaults to None.
        logposterior_values (np.ndarray, optional): logPosterior values. Defaults to None.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float64.
        Nbatches (int, optional): Number of batches. Defaults to 100.
        split_ratio (float, optional): Ratio to split data into training and validation sets. Defaults to 0.8.
        conversion_method (str, optional): Method for data conversion to the flow latent space ('normalize' or 'standardize'). Defaults to 'normalize'.
    
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
                 num_flow_steps: int = 5, 
                 use_nvp: bool = False, 
                 device: str | torch.device = 'cpu', 
                 verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 Nbatches: int = 100,
                 split_ratio: float = 0.8,
                 conversion_method: str = 'normalize'
                 ):
        
        
        self.split_ratio = split_ratio
        self._setup_conversions(conversion_method)
        self.posterior_samples = self._process_posterior_samples(posterior_samples)
        self._process_tensors()

        num_samples, num_dims = self.posterior_samples.shape
        self.Nbatches = Nbatches if Nbatches else num_samples
        self.batch_size = self.Nsamples // self.Nbatches
        self.logposterior_values = logposterior_values

        super().__init__(num_dims, num_flow_steps, use_nvp, device, dtype, verbose)

    def _setup_conversions(self, conversion_method):
        """
        Sets up the conversion methods for transforming data to and from latent space.
        
        Args:
            conversion_method (str): The method to use for conversion. Must be one of 'normalize' or 'standardize'.
        
        Raises:
            ValueError: If an invalid conversion method is provided.
        """

        if conversion_method == 'normalize':
            self._to_latent_space = normalize
            self._from_latent_space = denormalize
        elif conversion_method == 'standardize':    
            self._to_latent_space = standardize
            self._from_latent_space = destandardize
        else:
            raise ValueError(f"Invalid conversion method: {conversion_method}. Choose from 'normalize' or 'standardize'.")
    
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
        6. Stores the latent target as a NumPy array.

        Attributes:
            self.q1: The first component of the latent space representation.
            self.q2: The second component of the latent space representation.
            self.latent_target: The latent space representation of the posterior samples as a NumPy array.
        """
        
        latent_samples, self.q1, self.q2 = self._to_latent_space(self.posterior_samples)
        shuffled_samples = shuffle(latent_samples)
        if self.split_ratio:
            train_samples, val_samples = split(shuffled_samples, self.split_ratio)
        else:
            train_samples, val_samples = shuffled_samples, None

        train_loader, val_loader = create_data_loaders(train_samples, val_samples, batch_size=self.batch_size, num_workers=4)
        self.load_data(train_loader, val_loader)

        self.latent_target = latent_samples.cpu().detach().numpy()

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
            float: The mean log evidence.
            float: The standard deviation of the log evidence.
        """
        self._sanity_check_flow(load_kwargs, train_kwargs)
        
        logProb = self.flow.log_prob(self.latent_target.to(self.device)).cpu().detach().numpy()
        logZ = self.logposterior_values - logProb
        mean, std = np.mean(logZ), np.std(logZ)
        logging.debug(f"LogZ: {mean} +/- {std}")
        return mean, std
    
    def get_draws(self, load_kwargs: dict = {}, train_kwargs: dict = {}, num_draws: int = 10000):
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
        samples = self.flow.sample(num_draws).cpu().detach().numpy()

        return self._from_latent_space(samples, self.q1, self.q2)
    
    def _sanity_check_flow(self, load_kwargs: dict = {}, train_kwargs: dict = {}):
        """
        Checks if the flow model is loaded or trained and loads or trains it if necessary.
        
        Args:
            load_kwargs (dict): Keyword arguments for loading the flow model. Refer to the documentation for the `load` method.
            train_kwargs (dict): Keyword arguments for training the flow model. Refer to the documentation for the `train` method.
        """
        if not hasattr(self, 'flow'):
            logging.debug("Building flow")
            self.build_flow()
        
        load = self.load(**load_kwargs)
        
        if not load:
            train_kwargs['target_distribution'] = self.latent_target
            logging.debug("Training flow")
            self.train(**train_kwargs)

class ErynEvidenceFlow(EvidenceFlow):
    """
    Wrapper class for using the ``EvidenceFlow`` class directly with a backend from the ``Eryn`` mcmc sampler.
    
    Args:
        backend (str or HDFBackend): The backend to load the samples from.
        loader (DataLoader): The DataLoader object to load the samples from.
        ess (int): The effective sample size. Default is 1e4.
        num_flow_steps (int): Number of flow steps in the model. Default is 5.
        use_nvp (bool): Whether to use NVP architecture. Default is False.
        device (str or torch.device): Device to run the model on. Default is 'cpu'.
        verbose (bool): Whether to print verbose output during training. Default is False.
        dtype (torch.dtype): Data type for tensors. Default is torch.float64.
        Nbatches (int): Number of batches. Default is 100.
        split_ratio (float): Ratio to split data into training and validation sets. Default is 0.8.
        conversion_method (str): Method for data conversion to the flow latent space ('normalize' or 'standardize'). Default is 'normalize'.
    """

    def __init__(self,
                backend: str | HDFBackend = None,
                loader: pysco.eryn.DataLoader = None,
                ess: int = int(1e4),
                num_flow_steps: int = 5, 
                use_nvp: bool = False, 
                device: str | torch.device = 'cpu', 
                verbose: bool = False,
                dtype: torch.dtype = torch.float64,
                Nbatches: int = 100,
                split_ratio: float = 0.8,
                conversion_method: str = 'normalize'):
        
        
        
        if not eryn_here:
            raise ImportError("Eryn is not installed. Please install Eryn to use this class, or \
                              use the EvidenceFlow class instead.")
        
        if backend is None and loader is None:
            raise ValueError("Either a backend or a loader must be provided.")
    
        elif loader is None and backend is not None:
            if pysco_here:
                loader = pysco.eryn.DataLoader(backend)
                samples, logL, logP = loader.load(ess=ess, squeeze=True)
            else:
                if isinstance(backend, str):
                    backend = HDFBackend(backend)

                samples, logP = self._load_samples_posterior(backend, ess)
        
        else:
            samples, logL, logP = loader.load(ess=ess, squeeze=True)

        super().__init__(posterior_samples=samples, 
                         logposterior_values=logP, 
                         num_flow_steps=num_flow_steps, 
                         use_nvp=use_nvp, 
                         device=device, 
                         verbose=verbose, 
                         dtype=dtype, 
                         Nbatches=Nbatches, 
                         split_ratio=split_ratio, 
                         conversion_method=conversion_method)
        
    def _compute_discard_thin(self, samples, ess=int(1e4)):
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
    
    def _load_samples_posterior(self, backend, ess=None):
        """
        Load the samples from the backend. If the effective sample size is provided, the number of samples to discard and thin is computed.
        This is NOT compatible with reversible jump MCMC yet.

        Args:
            backend (HDFBackend): The backend to load the samples from.
            ess (int, optional): The effective sample size. Defaults to None.

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
            samples_out[name] = np.squeeze(samples[name][discard::thin, 0]).reshape(-1, nd) #take the first temperature chain and flatten the rest
        
        logP = backend.get_log_posterior(discard=discard, thin=thin)[:, 0].flatten()

        return samples_out, logP
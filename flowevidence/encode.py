# coding: utf-8
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
import logging
from .utils import EarlyStopping, setup_logging, lossplot, cornerplot_training, clean_chain
from tqdm import tqdm


class MaskedEncoder(nn.Module):
    """
    Encoder to handle variable-dimension data (e.g., RJ-MCMC branches).

    Args:
        max_model_dim (int): Maximum dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        hidden_dim (int, optional): Hidden dimension for the encoder. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        use_vae (bool, optional): If True, use a variational autoencoder. Defaults to False.
        device (str | torch.device, optional): Device to use for training. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float64.
    """
    def __init__(self, 
                 max_model_dim: int, 
                 latent_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 use_vae: bool = False,
                 device: str | torch.device = 'cpu',
                 dtype: torch.dtype = torch.float64
                 ):
        
        super().__init__()

        self.fc1 = nn.Linear(2*max_model_dim, hidden_dim, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(hidden_dim, device=device, dtype=dtype) 

        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(2*hidden_dim, device=device, dtype=dtype)

        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm1d(4*hidden_dim, device=device, dtype=dtype)

        self.fc4 = nn.Linear(4*hidden_dim, 2*hidden_dim, device=device, dtype=dtype)
        self.bn4 = nn.BatchNorm1d(2*hidden_dim, device=device, dtype=dtype)
        if use_vae:
            self.fc5_mu = nn.Linear(2*hidden_dim, latent_dim, device=device, dtype=dtype)
            self.fc5_logvar = nn.Linear(2*hidden_dim, latent_dim, device=device, dtype=dtype)
        else:
            self.fc5 = nn.Linear(2*hidden_dim, latent_dim, device=device, dtype=dtype)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer

        self.forward = self.forward_vae if use_vae else self.forward_det

    def forward_det(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes input data into a latent space, considering the mask for valid data.

        Args:
            x (torch.Tensor): Input data of shape [N_samples, max_model_dim].
            mask (torch.Tensor): Mask indicating valid dimensions (1 = valid, 0 = invalid).

        Returns:
            torch.Tensor: Encoded data of shape [N_samples, latent_dim].
        """
        combined = torch.cat((x.nan_to_num(0.0), mask), dim=1)  # Replace NaNs in x with 0 and concatenate with mask
        z = self.fc1(combined)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.fc2(z)
        z = self.bn2(z)
        z = self.relu(z)
        #z = self.dropout(z)

        z = self.fc3(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.fc4(z)
        z = self.bn4(z)
        z = self.relu(z)
        #z = self.dropout(z)

        z = self.fc5(z)
        return z
    
    def forward_vae(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input data into a latent space, considering the mask for valid data. 

        Args:
            x (torch.Tensor): Input data of shape [N_samples, max_model_dim].
            mask (torch.Tensor): Mask indicating valid dimensions (1 = valid, 0 = invalid).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded data mean and log-variance of the latent space.
        """
        combined = torch.cat((x.nan_to_num(0.0), mask), dim=1)
        z = self.fc1(combined)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.dropout(z)
        
        z = self.fc2(z)
        z = self.bn2(z)
        z = self.relu(z)
        #z = self.dropout(z)

        z = self.fc3(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.fc4(z)
        z = self.bn4(z)
        z = self.relu(z)
        #z = self.dropout(z)

        mu = self.fc5_mu(z)
        logvar = self.fc5_logvar(z)

        return mu, logvar

class MaskedDecoder(nn.Module):
    """
    Decoder for reconstructing data from the latent space.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        max_model_dim (int): Maximum dimensionality of the output (original data space).
        hidden_dim (int, optional): Hidden dimension for the decoder. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        use_vae (bool, optional): If True, use a variational autoencoder. Defaults to False.
        device (str | torch.device, optional): Device to use for training. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float64.
    """
    def __init__(self, 
                 latent_dim: int, 
                 max_model_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 use_vae: bool = False,
                 device: str | torch.device = 'cpu',
                 dtype: torch.dtype = torch.float64
                 ):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.max_model_dim = max_model_dim
           
        self.fc1 = nn.Linear(latent_dim, hidden_dim, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(hidden_dim, device=device, dtype=dtype)

        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(2*hidden_dim, device=device, dtype=dtype)

        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm1d(4*hidden_dim, device=device, dtype=dtype)

        self.fc4 = nn.Linear(4*hidden_dim, 2*hidden_dim, device=device, dtype=dtype)
        self.bn4 = nn.BatchNorm1d(2*hidden_dim, device=device, dtype=dtype)
        
        self.fc5 = nn.Linear(2*hidden_dim, 2*max_model_dim, device=device, dtype=dtype)

        if use_vae:
            self.out = torch.tanh
        else:
            self.out = lambda x: x

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) # Dropout layer

    def forward(self, 
                z: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input latent representation (batch_size, latent_dim).
        
        Returns:
            torch.Tensor: Reconstructed data (batch_size, max_model_dim).
        """
        #return self.decoder(z)
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        #x = self.dropout(x)

        x = self.fc5(x)
        #x = self.out(x)

        reconstructed_data = x[:, :x.shape[1]//2]  # First half of the output is the reconstructed data
        reconstructed_mask = torch.sigmoid(x[:, x.shape[1]//2:])  # Use sigmoid for mask probabilities
        return reconstructed_data, reconstructed_mask


class MaskedAutoEncoder:
    """
    An autoencoder designed to convert samples into a latent space. The model can handle variable-dimension data, 
    such as RJ-MCMC branches, by using a mask to indicate "missing" entries at each step.
    This allows the compression of the RJ-MCMC samples into a fixed-size latent space that can be used to train
    the Flow used for the evidence calculation.

    Args:
        max_model_dim (int): Maximum dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        device (str | torch.device, optional): Device to use for training. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float64.
        use_vae (bool, optional): If True, use a variational autoencoder. Defaults to False.
        hidden_dim (int, optional): Hidden dimension for the encoder and decoder. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        verbose (bool, optional): If True, print training progress. Defaults to False.
    """
    def __init__(self, 
                max_model_dim: int, 
                latent_dim: int,
                device: str | torch.device = 'cpu',
                dtype: torch.dtype = torch.float64,
                use_vae: bool = False,
                hidden_dim: int = 128,
                dropout: float = 0.2,  
                verbose: bool = False
                ):
        
        if use_vae:
            logging.warning("Using a Variational Autoencoder. This is experimental and may not work as expected. We recommend using a deterministic autoencoder.")

        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        setup_logging(verbose)

        self.encoder = MaskedEncoder(max_model_dim, latent_dim, use_vae=use_vae, hidden_dim=hidden_dim, dropout=dropout, device=device, dtype=dtype)
        self.decoder = MaskedDecoder(latent_dim, max_model_dim, use_vae=use_vae, hidden_dim=hidden_dim, dropout=dropout, device=device, dtype=dtype)
        
        self.max_model_dim = max_model_dim
        self.loss_fn = use_vae
        self.get_latent = use_vae

        self.trained = False

    @property
    def loss_fn(self):
        return self._loss_fn
    
    @loss_fn.setter
    def loss_fn(self, use_vae: bool = False):
        if use_vae:
            self._loss_fn = self.VAE_loss_fn
        else:
            self._loss_fn = self.reconstruction_loss_fn

    @property
    def get_latent(self):
        return self._get_latent
    
    @get_latent.setter
    def get_latent(self, use_vae: bool = False):
        if use_vae:
            self._get_latent = self.get_z_vae
        else:
            self._get_latent = self.get_z_det

    def VAE_loss_fn(self, input, reconstruction, input_mask, reconstructed_mask, mean, logvar):

        # KL Divergence loss
        loss = self.reconstruction_loss_fn(input, reconstruction, input_mask, reconstructed_mask)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return loss + kl_loss

    def reconstruction_loss_fn(self, input, reconstruction, input_mask, reconstructed_mask, mean=None, logvar=None):
        """
        Computes the combined reconstruction loss for values and mask.

        Args:
            input (torch.Tensor): Original input data of shape [N_samples, max_model_dim].
            reconstruction (torch.Tensor): Reconstructed data of shape [N_samples, max_model_dim].
            input_mask (torch.Tensor): Original NaN mask of shape [N_samples, max_model_dim].
            reconstructed_mask (torch.Tensor): Reconstructed NaN mask.
            mean (torch.Tensor, optional): Mean of the latent space. Defaults to None. It is ignored unless using a VAE.
            logvar (torch.Tensor, optional): Log-variance of the latent space. Defaults to None. It is ignored unless using a VAE.

        Returns:
            loss (torch.Tensor): Combined reconstruction loss.
        """
        # Mask for valid entries
        diff = torch.nan_to_num(input, nan=0.0) - torch.nan_to_num(reconstruction, nan=0.0)
        #valid_loss = diff ** 2 MSE
        valid_loss = torch.log(torch.cosh(diff)) # Huber loss
        valid_loss = valid_loss.sum() / input_mask.sum()
        
        # Binary cross-entropy for the NaN mask reconstruction
        mask_loss = nn.functional.binary_cross_entropy(reconstructed_mask, input_mask)

        #breakpoint()

        return valid_loss + mask_loss

        
    def train(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_tensor: torch.Tensor=None,
            start_epoch: int = 0,
            epochs: int = 1000,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            lambda_L1: float = 0.0,
            early_stopping: bool | Callable = True,
            stopping_kwargs: Optional[dict] = {},
            path: str = './',
            filename: str = 'autoencoder.pth',):
        
        """
        Train the autoencoder model.

        Args:
            start_epoch (int, optional): The epoch to start training from. Defaults to 0.
            epochs (int, optional): The number of epochs to train for. Defaults to 1000.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): L2 regularization strength. Defaults to 0.0.
            early_stopping (bool | Callable, optional): If True, use early stopping with default parameters.
                If a callable is provided, it will be used as the early stopping function. Defaults to False.
            stopping_kwargs (Optional[dict], optional): Additional arguments for the early stopping function. Defaults to {}.
            path (str, optional): The directory path to save the model and diagnostics. Defaults to './'.
            filename (str, optional): The filename to save the trained model. Defaults to 'autoencoder.pth'.
        """        
        if test_tensor is not None:
            self.test_tensor = test_tensor.to(self.device)
            self.test_array = clean_chain(test_tensor.cpu().detach().numpy())
        else:
            self.test_tensor = None
            self.test_array = None

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
        
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr, weight_decay=weight_decay)
        if val_loader:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                factor=0.5,
                                                                patience=50,
                                                                threshold=1e-5)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        current_lr = lr

        # use tqdm for progress bar only if verbose is True
        epoch_iterator = tqdm(range(start_epoch, epochs), desc="Training", disable=not self.verbose)

        for epoch in epoch_iterator:
            
            train_loss = self._train_one_epoch(train_loader=train_loader, optimizer=optimizer, lambda_L1=lambda_L1)
            val_loss = self._validate_one_epoch(val_loader=val_loader, lambda_L1=lambda_L1) if val_loader else None

            scheduler.step(val_loss) if val_loader else scheduler.step()

            if stopping_fn:
                if stopping_fn(val_loss):
                    logging.info(f"Early stopping at epoch {epoch}")
                    converged = True
                    break

            if epoch  > 0 and epoch % 100 == 0:
                if self.verbose:
                    self._log_epoch(epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, savepath)
                    if scheduler.get_last_lr()[0] != current_lr:
                        current_lr = scheduler.get_last_lr()[0]
                        logging.info(f"New learning rate: {scheduler.get_last_lr()[0]}")
                    logging.info("Saving model @ epoch {}".format(epoch))

                self._save_model(trainedpath)
            
        if stopping_fn and not converged:
            logging.warning("Early stopping did not trigger")
        
        self.trained = True
        self._save_model(trainedpath)
            
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def get_z_vae(self, x, mask):
        mean, logvar = self.encoder(x, mask)
        z = self.reparameterize(mean, logvar)

        return z, mean, logvar

    def get_z_det(self, x, mask):
        z = self.encoder(x, mask)
        return z, None, None

    def _train_one_epoch(self, train_loader, optimizer, lambda_L1):
        """
        Perform a training step for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            lambda_L1 (float): L1 regularization strength.

        Returns:
            float: The average training loss for the epoch.
        """

        self.encoder.train()
        self.decoder.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch[0].to(self.device, non_blocking=self.device.type == 'cuda')
            mask = torch.isfinite(batch).to(self.dtype).to(self.device)

            latent, mean, logvar = self.get_latent(batch, mask)
            
            # Decode the latent representation
            reconstructed_data, reconstructed_mask = self.decoder(latent)

            # Reconstruction loss
            loss = self.loss_fn(batch, reconstructed_data, mask, reconstructed_mask, mean, logvar)

            #compute L1 regularization
            L1_penalty = lambda_L1 * torch.norm(latent, p=1)

            loss += L1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        return train_loss / len(train_loader)

    def _validate_one_epoch(self, val_loader, lambda_L1):
        """
        Perform a validation step for one epoch.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            lambda_L1 (float): L1 regularization strength.

        Returns:
            float: The average validation loss for the epoch.
        """
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_loader:
                batch = batch[0].to(self.device, non_blocking=self.device.type == 'cuda')
                mask = torch.isfinite(batch).to(self.dtype).to(self.device)

                latent, mean, logvar = self.get_latent(batch, mask)
                
                # Decode the latent representation
                reconstructed_data, reconstructed_mask = self.decoder(latent)

                # Reconstruction loss
                loss = self.loss_fn(batch, reconstructed_data, mask, reconstructed_mask, mean, logvar)

                #compute L1 regularization
                L1_penalty = lambda_L1 * torch.norm(latent, p=1)
                loss += L1_penalty

                val_loss += loss.item()

        return val_loss / len(val_loader)
    
    def _log_epoch(self, epoch, train_loss, val_loss, epochs_losses, train_losses, val_losses, savepath, ndim=15):
        """
        Logs the training and validation loss for a given epoch and updates the loss lists.

        Args:
            epoch (int): The current epoch number.
            train_loss (float): The training loss for the current epoch.
            val_loss (float or None): The validation loss for the current epoch, or None if not applicable.
            epochs_losses (list): A list to store the epoch numbers.
            train_losses (list): A list to store the training losses.
            val_losses (list): A list to store the validation losses.
            savepath (str): The directory path where the loss plot will be saved.
        """

        if val_loss is not None:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        else:
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss}')

        epochs_losses.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lossplot(epochs_losses, train_losses, val_losses, plot_dir=savepath, savename='autoencoder_loss')

        if self.test_tensor is not None:
            encoded = self.encode(self.test_tensor.reshape(-1, self.max_model_dim))
            decoded = self.decode(encoded)
            decoded_array = decoded.cpu().detach().numpy()
            decoded_array = decoded_array.reshape(-1, self.test_array.shape[1])

            logging.info('nans predicted by the autoencoder: %.i' % torch.isnan(decoded).sum().item())
            logging.info('nans present in the target: %.i' % torch.isnan(self.test_tensor).sum().item())    
            
            try:
                decoded_array = clean_chain(decoded_array)
                cornerplot_training(samples=decoded_array[:, :ndim], target_distribution=self.test_array[:, :ndim], epoch=epoch, plot_dir=savepath, savename='autoencoder_cornerplot')

            except ValueError as e:
                logging.info('Corner plot not generated: {} Resume training'.format(e))

    def _save_model(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'trained': self.trained,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """
        Load a saved model from a file.

        Args:
            path (str): Path to the saved model.
        """
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.trained = checkpoint['trained']
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode data into the latent space.

        Args:
            data (torch.Tensor): Input data of shape [N_samples, max_model_dim].

        Returns:
            torch.Tensor: Encoded data of shape [N_samples, latent_dim].
        """
        mask = torch.isfinite(data).to(self.dtype).to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            latent, _, _ = self.get_latent(data, mask)
            return latent
    
    def decode(self, latent: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Decode data from the latent space.

        Args:
            latent (torch.Tensor): Input latent representation of shape [N_samples, latent_dim].
            threshold (float): Threshold to classify mask probabilities as valid or NaN.

        Returns:
            torch.Tensor: Decoded data of shape [N_samples, max_model_dim].
        """
        self.decoder.eval()
        with torch.no_grad():
            reconstructed_data, reconstructed_mask =  self.decoder(latent)
        
        return self.postprocess_decoder_output(reconstructed_data, reconstructed_mask, threshold)

    def postprocess_decoder_output(self, reconstructed_data, reconstructed_mask, threshold=0.5):
        """
        Post-process the decoder output to reintroduce NaNs where necessary.

        Args:
            reconstructed_data (torch.Tensor): Reconstructed data of shape [N_samples, max_model_dim].
            reconstructed_mask (torch.Tensor): Reconstructed NaN mask.
            threshold (float): Threshold to classify mask probabilities as valid or NaN.

        Returns:
            reconstructed_data (torch.Tensor): Post-processed data with NaNs reintroduced.
        """
        nan_positions = (reconstructed_mask < threshold)
        reconstructed_data[nan_positions] = float('nan')  # Replace positions with NaN
        return reconstructed_data

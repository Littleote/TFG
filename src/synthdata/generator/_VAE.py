# -*- coding: utf-8 -*-
"""
Variational AutoEncoder

@author: david
"""

import numpy as np
import torch
import optuna
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import time as gettime

from ._base import Generator, if_fitted, fit_method

def loss_function(recon_x, x, mu, logvar):
    recLoss = nn.MSELoss(reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
    return recLoss(recon_x, x) + torch.sqrt(KLD)

class VAE(Generator, nn.Module):
    def __init__(self, device: 'str | device' = 'auto',
                 layers_sizes: 'list | None' = None, 
                 latent_dimension: 'int | None' = None, layers: 'int | None' = None,
                 epochs: 'int | None' = None, search_trials: 'int' = 10,
                 batches: 'int | None' = None, batch_size: 'int | None' = 128,
                 learning_rate: 'float' = 1e-1,
                 reduce_learning_rate: 'bool' = True, min_learning_rate: 'float' = 1e-5,
                 patience: 'int | None' = None, time_limit: 'float' = 10):
        super().__init__()
        if device == 'auto':
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif type(device) == str:
            self._device = torch.device(device)
        else:
            self._device = device
        if layers_sizes is None:
            self.layers_sizes = None
            self.latent_dimension = latent_dimension
            self.layers = layers
        else:
            self.layers_sizes = layers_sizes
            self.latent_dimension = layers_sizes[-1]
            self.layers = len(layers_sizes)
        self.epochs = epochs
        self.search_trials = search_trials
        self.batches = batches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reduce_learning_rate = reduce_learning_rate
        self.min_learning_rate = min_learning_rate
        self.patience = patience
        self.time_limit = time_limit
    
    @fit_method
    def fit(self, X):
        torch_X = torch.from_numpy(X).float().to(self._device)
        self.n, self.dim = X.shape
        
        if self.layers_sizes is None:
            params = {
                'latent_dimension': self.latent_dimension,
                'layers': self.layers
            }
            
            if self.latent_dimension is None or self.layers is None:
                optuna.logging.set_verbosity(optuna.logging.ERROR)
                study = optuna.create_study()
                study.optimize(self._get_objective(torch_X), n_trials=self.search_trials)
                params = params | study.best_params
            latent_dimension = params['latent_dimension']
            layers = params['layers']
            
            layers_sizes = np.array(np.round(np.exp(np.linspace(np.log(self.dim), np.log(latent_dimension), layers + 1))), dtype=int)
        else:
            layers_sizes = [self.dim] + self.layers_sizes
        
        return self._run_configuration(torch_X, layers_sizes)
        
    @if_fitted
    def generate(self, size):
        with torch.no_grad():
            S = torch.zeros((size, self.used_latent_dimension), device=self._device, dtype=torch.float).normal_()
            X = self.decode(S)
        return X.detach().cpu().numpy().astype(float)
    
    # VAE forward steps
    def encode(self, x):
        enc = self.encoder(x)
        return self.mean(enc), self.logvar(enc)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.zeros(std.size(), device=self._device.type, dtype=torch.float).normal_()
        return mean + eps * std

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        return self.decode(z), mean, logvar
    
    # VAE training
    def _train(self, X):
        time_limit = self.time_limit + gettime()
        batches = max(round(self.n / self.batch_size), 1) if self.batches is None else self.batches
        batch_size = int(np.ceil(self.n / batches))
        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        loop = True
        if self.epochs is not None:
            mode = 'epochs'
            current_epoch = 0
            if self.reduce_learning_rate:
                patience = 10 if self.patience is None else self.patience
                schedule = ReduceLROnPlateau(optimizer, patience=patience)
        elif self.reduce_learning_rate:
            mode = 'lr'
            patience = 5 if self.patience is None else self.patience
            schedule = ReduceLROnPlateau(optimizer, patience=patience)
        else:
            mode = 'patience'
            patience = 20 if self.patience is None else self.patience
            best_loss = None
            since_improvement = 0
            
        self.train()
        while loop:
            loss = 0
            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                
                outputs, mean, logvar = self.forward(batch)
                
                train_loss = loss_function(outputs, batch, mean, logvar)
                train_loss.backward()
                
                optimizer.step()
                
                loss += train_loss.detach().cpu()
            if self.reduce_learning_rate:
                schedule.step(loss)
                
            
            if mode == 'epochs':
                current_epoch += 1
                loop = current_epoch < self.epochs
            elif mode == 'lr':
                loop = optimizer.state_dict()['param_groups'][0]['lr'] > self.min_learning_rate
            else:
                since_improvement += 1
                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    since_improvement = 0
                loop = since_improvement < patience
            loop &= time_limit > gettime()
        self.eval()
    
    def _run_configuration(self, X, layers_sizes):
        latent_dimension = layers_sizes[-1]
        self.used_latent_dimension = latent_dimension
        
        f = nn.Sigmoid()
        
        self.encoder = nn.Sequential(*[
            f if i == -1 else nn.Linear(layers_sizes[i], layers_sizes[i + 1])
            for i in [e // 2 if e % 2 == 0 else -1 for e in range(2 * len(layers_sizes) - 4)]
            ])
        self.mean = nn.Linear(layers_sizes[-2], latent_dimension)
        self.logvar = nn.Linear(layers_sizes[-2], latent_dimension)
        self.decoder = nn.Sequential(*[
            f if i == -1 else nn.Linear(layers_sizes[i], layers_sizes[i - 1])
            for i in [e // 2 if e % 2 == 0 else -1 for e in range(2 * len(layers_sizes) - 2, 1, -1)]
            ])
        
        self.to(self._device)
        self._train(X)
        with torch.no_grad():
            recLoss = nn.MSELoss(reduction='mean')
            mean, logvar = self.encode(X)
            Y = self.decode(mean)
            loss = recLoss(X, Y)
            # noise lost because of dimensionality reduction
            # Noise = {Uncorrelated variance} * {Average distance to 0 of N(0, 1)} * {#Lost dimensions}
            noise = 0.05 * np.sqrt(2 / np.pi) * (self.dim - latent_dimension)
        return loss - noise
    
    def _get_objective(self, X):
        def objective(trial):
            latent_dimension = self.latent_dimension
            layers = self.layers
            latent_dimension = trial.suggest_int('latent_dimension', 1, self.dim) \
                            if latent_dimension is None \
                            else latent_dimension
            layers = trial.suggest_int('layers', 1, 5) if layers is None else layers
            
            layers_sizes = np.array(np.round(np.exp(np.linspace(np.log(self.dim), np.log(latent_dimension), layers + 1))), dtype=int)
            
            return self._run_configuration(X, layers_sizes)
        return objective
    
    def prefit(self, X):
        self.n, self.dim = X.shape
        torch_X = torch.from_numpy(X).float().to(self._device)
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study()
        study.optimize(self._get_objective(torch_X), n_trials=self.search_trials)
        
        params = {
            'latent_dimension': self.latent_dimension,
            'layers': self.layers
        } | study.best_params
        self.latent_dimension = params['latent_dimension']
        self.layers = params['layers']
        
        # Clean neural network
        self.encoder = None
        self.mean = None
        self.logvar = None
        self.decoder = None
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

from ._base import BaseGenerator

def default_loss_function(recon_x, x, mu, logvar):
    recLoss = nn.MSELoss(reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
    return recLoss(recon_x, x) + torch.sqrt(KLD)

class VAE(BaseGenerator, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def encode(self, x):
        enc = self.encoder(x)
        return self.mean(enc), self.logvar(enc)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        if torch.cuda.is_available() and False:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return mean + eps * std

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        return self.decode(z), mean, logvar
    
    def _train(self, X, epochs, criterion=default_loss_function, batches=None, batch_size=64, lr=1e-1, reduce_lr=True, patience=10, max_time=10):
        batches = round(self.n / batch_size) if batches is None else batches
        batch_size = int(np.ceil(self.n / batches))
        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        schedule = ReduceLROnPlateau(optimizer, patience=patience)
        self.train()
        loop = True
        epoch = 0
        best = None
        since = 0
        time = max_time + gettime()
        losses = []
        while loop:
            loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs, mean, logvar = self.forward(batch)
                
                train_loss = criterion(outputs, batch, mean, logvar)
                train_loss.backward()
                
                optimizer.step()
                
                loss += train_loss.item()
            if reduce_lr:
                schedule.step(loss)
                
            if not epochs == 'auto':
                epoch += 1
                loop = epoch < epochs
            elif reduce_lr:
                loop = optimizer.state_dict()['param_groups'][0]['lr'] > 1e-7
            else:
                since += 1
                if best is None or best > loss:
                    best = loss
                    since = 0
                loop = since < patience
            loop &= time > gettime()
            losses.append(loss)
        self.eval()
        # import matplotlib.pyplot as plt
        # plt.plot(losses)
        # plt.show()
        
    def probabilities(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        tX = torch.from_numpy(X)
        with torch.no_grad():
            mX, _ = self.encode(tX)
            dist = torch.sum(torch.square(mX), 1)
            probs = torch.exp(-dist / 2) / np.pow(np.sqrt(2 * np.pi), self.enc_dim)
        return probs.numpy()
    
    def setup_run(self, X, phases, **args):
        self.enc_dim = phases[-1]
        
        f = nn.Sigmoid()
        recLoss = nn.MSELoss(reduction='sum')
        
        self.encoder = nn.Sequential(*[
            f if i == -1 else nn.Linear(phases[i], phases[i + 1])
            for i in [e // 2 if e % 2 == 0 else -1 for e in range(2 * len(phases) - 4)]
            ])
        self.mean = nn.Linear(phases[-2], self.enc_dim)
        self.logvar = nn.Linear(phases[-2], self.enc_dim)
        self.decoder = nn.Sequential(*[
            f if i == -1 else nn.Linear(phases[i], phases[i - 1])
            for i in [e // 2 if e % 2 == 0 else -1 for e in range(2 * len(phases) - 2, 1, -1)]
            ])
        
        self._train(X, **args)
        with torch.no_grad():
            Y, _, _ = self.forward(X)
        return recLoss(X, Y)
        
    
    def _fit(self, X, phases='auto', enc_dim='auto', layers='auto', epochs='auto', **train_args):
        tX = torch.from_numpy(X).float()
        self.n, self.dim = X.shape
        
        if phases == 'auto':
            self._optimze_args = {'enc_dim': enc_dim, 'layers': layers}
            
            if enc_dim == 'auto' or layers == 'auto':
                def objective(trial):
                    enc_dim = self._optimze_args['enc_dim']
                    layers = self._optimze_args['layers']
                    enc_dim = trial.suggest_int('enc_dim', 1, self.dim) if enc_dim == 'auto' else enc_dim
                    layers = trial.suggest_int('layers', 1, 2 * self.dim) if layers == 'auto' else layers
                    
                    phases = np.array(np.round(np.exp(np.linspace(np.log(self.dim), np.log(enc_dim), layers + 1))), dtype=int)
                    
                    return self.setup_run(tX, phases, epochs=epochs, **train_args)
                
                study = optuna.create_study()
                study.optimize(objective, n_trials=10)
                
                self._optimze_args = study.best_params
            enc_dim = self._optimze_args['enc_dim']
            layers = self._optimze_args['layers']
            
            phases = np.array(np.round(np.exp(np.linspace(np.log(self.dim), np.log(enc_dim), layers + 1))), dtype=int)
            
        return self.setup_run(tX, phases, epochs=epochs, **train_args)
            
    def _generate(self, size):
        with torch.no_grad():
            S = torch.zeros((size, self.enc_dim), dtype=torch.float).normal_()
            X = self.decode(S)
        return X.numpy()